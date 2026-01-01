# ==============================================================================
# 03_prep_split_scale.R (H2)
#
# PURPOSE:
#   Prepare the final feature set for modeling by:
#     - Engineering UPB and modification features
#     - Deriving target variables for each snapshot
#     - Splitting into train/test sets
#     - Calculating scaling parameters for numeric features
#   Ensures all artifacts are consistent with H1 pipeline requirements.
#
# INPUTS:
#   - final_features_raw_clean.rds
# OUTPUTS:
#   - preproc_params.rds
#   - train_targets.rds
#   - test_targets.rds
#   - final_data_base_with_targets.rds
# ==============================================================================

# --- 0. DEPENDENCIES ---
library(data.table)
library(lubridate)
source("00_config.R")

# --- 1. LOAD CLEANED FINAL FEATURES -----------------------------------------
FINAL_FEATURES_FILE <- file.path(SAVE_DIR, "final_features_raw_clean.rds")
final_data <- readRDS(FINAL_FEATURES_FILE)
setDT(final_data)

# --- 1B. FEATURE ENGINEERING -------------------------------------------------
cat("--- 1B. Engineering UPB & Modification Features ---\n")

# 1. Compute UPB Percentage (Normalized Equity Proxy)
if (all(c("current_upb", "orig_upb") %in% names(final_data))) {
  final_data[, upb_pct_remaining := current_upb / orig_upb]
  final_data[is.na(upb_pct_remaining) | is.infinite(upb_pct_remaining), upb_pct_remaining := 0]
}

# 2. Modification Flag → binary feature
if ("Modification_Flag" %in% names(final_data)) {
  
  cat("   Processing Modification_Flag...\n")
  final_data[, is_modified := 0L]
  
  # Both 'Y' (current) and 'P' (prior) indicate modification
  final_data[trimws(as.character(Modification_Flag)) %in% c("Y", "P"), is_modified := 1L]
  
  # Drop original text column to avoid model issues
  final_data[, Modification_Flag := NULL]
  
} else {
  stop("CRITICAL ERROR: Modification_Flag column missing. Check 00_config.R mapping.")
}

# 3. Drop raw Loan_Age (use engineered clean_loan_age instead)
if ("Loan_Age" %in% names(final_data)) {
  cat("   Dropping raw Loan_Age (using clean_loan_age instead)...\n")
  final_data[, Loan_Age := NULL]
}

# --- 2. TARGET DERIVATION ---------------------------------------------------
cat("--- 2. Deriving Targets ---\n")

# Convert delinquency status to numeric: 0 = current, 1/2 = minor delinquency, 3+ = default
final_data[, delq_status_numeric := fcase(
  if_delq_sts == "RA", 3L,
  default = suppressWarnings(as.integer(if_delq_sts))
)]
final_data[is.na(delq_status_numeric), delq_status_numeric := 0L]

# Ensure Default_Month exists for all loans
if (!"Default_Month" %in% names(final_data)) {
  default_lookup <- final_data[delq_status_numeric >= 3,
                               .(Default_Month = min(Monthly_Reporting_Period)),
                               by = "Loan_Sequence_Number"]
  final_data[default_lookup, Default_Month := i.Default_Month, on = "Loan_Sequence_Number"]
  final_data[is.na(Default_Month), Default_Month := 999999L]
}

# Define all snapshot periods
possible_end_dates <- as.integer(format(
  seq(ymd(paste0(START_PERIOD_TRAIN, "01")), ymd(paste0(TEST_END_DATE, "01")), by = "1 month"), "%Y%m"
))

# Function to derive target for a single snapshot
derive_target_for_snapshot <- function(t) {
  horizon_start_dt <- ymd(paste0(t, "01")) %m+% months(1)
  horizon_end_dt   <- ymd(paste0(t, "01")) %m+% months(12)
  h_start <- as.integer(format(horizon_start_dt, "%Y%m"))
  h_end   <- as.integer(format(horizon_end_dt, "%Y%m"))
  
  # Observation window: prior 6 months to mask already defaulted loans
  observation_start_dt <- ymd(paste0(t, "01")) %m-% months(5) 
  obs_start <- as.integer(format(observation_start_dt, "%Y%m"))
  
  # Active loans for this snapshot
  active_loans <- final_data[
    Monthly_Reporting_Period == t & 
      (Default_Month > t | (Default_Month >= obs_start & Default_Month <= t)),
    .(Loan_Sequence_Number, Default_Month) 
  ]
  
  # Identify any defaults in prediction horizon
  horizon_defaults <- final_data[
    Loan_Sequence_Number %in% active_loans$Loan_Sequence_Number &
      Monthly_Reporting_Period >= h_start & 
      Monthly_Reporting_Period <= h_end &
      delq_status_numeric >= 3,
    .(Has_Default = TRUE),
    by = Loan_Sequence_Number
  ]
  
  # Merge defaults onto active loans
  target_results <- horizon_defaults[active_loans, on = "Loan_Sequence_Number"]
  
  # Assign target labels
  target_results[, Target_Y := fcase(
    Default_Month <= t, -1L,
    !is.na(Has_Default), 1L,
    default = 0L
  )]
  
  target_results[, Snapshot_Date := t]
  
  # Mask future loans beyond data horizon
  if (h_end > PERF_END_PERIOD) {
    target_results[Target_Y == 0, Target_Y := NA_integer_]
  }
  
  return(target_results[, .(Loan_Sequence_Number, Snapshot_Date, Target_Y)])
}

# Generate targets for all snapshots
all_targets_dt <- rbindlist(lapply(possible_end_dates, derive_target_for_snapshot), use.names = TRUE)
all_targets_dt <- all_targets_dt[!is.na(Target_Y)]

# Merge targets with features
final_data_with_targets <- merge(
  final_data, all_targets_dt,
  by.x = c("Loan_Sequence_Number", "Monthly_Reporting_Period"),
  by.y = c("Loan_Sequence_Number", "Snapshot_Date"),
  all.x = TRUE
)

# --- 3. SPLIT TRAIN / TEST --------------------------------------------------
train_targets <- all_targets_dt[Snapshot_Date <= TRAIN_SPLIT_DATE]
test_targets  <- all_targets_dt[Snapshot_Date > TRAIN_SPLIT_DATE]

# --- 4. SCALING PARAMETERS --------------------------------------------------
train_features <- final_data_with_targets[
  Monthly_Reporting_Period >= PERF_START_HISTORY &
    Monthly_Reporting_Period <= TRAIN_SPLIT_DATE
]

safe_stat <- function(x, fn) if (all(is.na(x))) NA_real_ else fn(x, na.rm = TRUE)
preproc_params <- data.table(Feature = character(), Min = numeric(), Max = numeric(), Median = numeric())
all_cols <- intersect(c(FINAL_ORIG_COLS_TO_SCALE, FINAL_PERF_COLS_TO_SCALE), names(train_features))

for (col in all_cols) {
  vals <- train_features[[col]]
  preproc_params <- rbindlist(list(preproc_params, list(
    Feature = col,
    Min = safe_stat(vals, min),
    Max = safe_stat(vals, max),
    Median = safe_stat(vals, median)
  )))
}

# --- 5. SAVE ARTIFACTS ------------------------------------------------------
cat("--- 5. Saving ---\n")

# Ensure Python pipeline reads numeric delinquency
if ("delq_status_numeric" %in% names(final_data_with_targets)) {
  final_data_with_targets[, if_delq_sts := delq_status_numeric]
  final_data_with_targets[, delq_status_numeric := NULL]
}

# Drop columns not needed for training
cols_to_drop <- c("Default_Month", "First_Pmt_Date") 

existing_drops <- intersect(cols_to_drop, names(final_data_with_targets))
if (length(existing_drops) > 0) {
  cat(sprintf("Dropping non-training columns: %s\n", paste(existing_drops, collapse=", ")))
  final_data_with_targets[, (existing_drops) := NULL]
}

# Save all artifacts
saveRDS(preproc_params,          file.path(SAVE_DIR, "preproc_params.rds"))
saveRDS(train_targets,           file.path(SAVE_DIR, "train_targets.rds"))
saveRDS(test_targets,            file.path(SAVE_DIR, "test_targets.rds"))
saveRDS(final_data_with_targets, file.path(SAVE_DIR, "final_data_base_with_targets.rds"))