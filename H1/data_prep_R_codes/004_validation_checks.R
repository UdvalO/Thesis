# ==============================================================================
# 004_validation_checks.R (H1)
#
# PURPOSE:
#   Comprehensive validation suite for H1 data pipeline outputs.
#   Checks cover:
#     - File existence & dimensions
#     - Train/test split and time window integrity
#     - Target variable distribution
#     - H1 feature correctness (raw UPB, binary delinquency)
#     - Graph construction readiness (Geo_Key / Lender_Key)
#
# INPUTS:
#   - 000_config.R
#   - orig_data_cleaned.rds (ORIG_SAVE_FILE)
#   - perf_data_cleaned.rds (PERF_SAVE_FILE)
#   - final_features_h1.rds (FINAL_FEATURES_FILE)
#   - train_targets.rds (TRAIN_TARGETS_FILE)
#   - test_targets.rds (TEST_TARGETS_FILE)
#   - preproc_params.rds (PREPROC_PARAMS_FILE)
#
# OUTPUTS:
#   - validation_summary_h1.csv
# ==============================================================================


# --- 0. DEPENDENCIES ---
library(data.table)
library(lubridate)
source("000_config.R")

cat("\n========================================\n")
cat("      DATA PIPELINE VALIDATION CHECKS (H1)\n")
cat("========================================\n\n")

# --- Initialize Validation Summary Table ------------------------------------
validation_summary <- data.table(
  Check_ID = character(),
  Check_Name = character(),
  Status = character(), # PASS, FAIL, WARN
  Metric = character(),
  Notes = character()
)

add_check <- function(id, name, status, metric = "", notes = "") {
  validation_summary <<- rbind(validation_summary, list(id, name, status, as.character(metric), notes))
}

# --- 1. Load All Required H1 Files -------------------------------------------
cat("--- 1. LOADING ALL SAVED FILES ---\n")
files_to_check <- list(
  orig_data = ORIG_SAVE_FILE,
  perf_data = PERF_SAVE_FILE,
  final_with_targets = FINAL_FEATURES_FILE,
  train_targets = TRAIN_TARGETS_FILE,
  test_targets = TEST_TARGETS_FILE,
  preproc_params = PREPROC_PARAMS_FILE
)

loaded_data <- list()
all_files_loaded <- TRUE

for (name in names(files_to_check)) {
  file_path <- files_to_check[[name]]
  if (file.exists(file_path)) {
    loaded_data[[name]] <- readRDS(file_path)
    if (is.data.frame(loaded_data[[name]])) setDT(loaded_data[[name]])
    cat(sprintf("%s loaded successfully\n", name))
  } else {
    cat(sprintf("%s NOT FOUND at %s\n", name, file_path))
    all_files_loaded <- FALSE
  }
}
cat("\n")
add_check("1.1", "File Existence", ifelse(all_files_loaded, "PASS", "FAIL"),
          sprintf("%d/%d files", length(loaded_data), length(files_to_check)), "All H1 artifacts must exist.")
if (!all_files_loaded) stop("Critical data files are missing. Halting validation.")

# --- 2. Sample Size & Dimension Checks ---------------------------------------
cat("--- 2. DIMENSION & SAMPLE SIZE CHECKS ---\n")
final_dt <- loaded_data$final_with_targets
n_unique_loans <- uniqueN(final_dt$Loan_Sequence_Number)
cat(sprintf("Unique Loans in Final Dataset: %d\n", n_unique_loans))
is_sample_size_correct <- n_unique_loans >= 200000
cat(ifelse(is_sample_size_correct, "Sample size matches H1 Target (250k scale)\n",
           "WARNING: Sample size is low (< 200k). Check data prep.\n"))
add_check("2.1", "Sample Size", ifelse(is_sample_size_correct, "PASS", "WARN"),
          sprintf("%d loans", n_unique_loans), "Target is ~250k.")

# --- 3. Train/Test Time Window Validation -----------------------------------
cat("--- 3. TIME WINDOW VALIDATION ---\n")
train_range <- range(loaded_data$train_targets$Snapshot_Date, na.rm = TRUE)
test_range <- range(loaded_data$test_targets$Snapshot_Date, na.rm = TRUE)
cat(sprintf("Training snapshots: %d to %d\n", train_range[1], train_range[2]))
cat(sprintf("Testing snapshots:  %d to %d\n", test_range[1], test_range[2]))
no_overlap <- train_range[2] < test_range[1]
add_check("3.1", "Train/Test Overlap", ifelse(no_overlap, "PASS", "FAIL"),
          sprintf("Train Ends %d, Test Starts %d", train_range[2], test_range[1]), "Strict temporal split.")

# --- 4. Target Distribution Check -------------------------------------------
cat("--- 4. TARGET DISTRIBUTION CHECK ---\n")
train_def <- mean(loaded_data$train_targets$Target_Y == 1, na.rm = TRUE)
test_def  <- mean(loaded_data$test_targets$Target_Y == 1, na.rm = TRUE)
cat(sprintf("Train Default Rate: %.2f%%\n", train_def * 100))
cat(sprintf("Test Default Rate:  %.2f%%\n", test_def * 100))
is_rate_similar <- abs(train_def - test_def) < 0.05
add_check("4.1", "Target Rate Stability", ifelse(is_rate_similar, "PASS", "WARN"),
          sprintf("Train: %.2f%%, Test: %.2f%%", train_def*100, test_def*100),
          "Default rates between train/test should be reasonably close.")

# --- 5. H1 Feature Correctness ----------------------------------------------
cat("\n--- 5. H1 FEATURE CORRECTNESS CHECK ---\n")

# 5A. Raw UPB vs Ratio Feature
has_raw_upb <- "current_upb" %in% names(final_dt)
has_upb_ratio <- "upb_pct_remaining" %in% names(final_dt)
if (has_raw_upb && !has_upb_ratio) {
  cat("Correct Feature Set: Raw UPB present, no ratio features.\n")
  status_feat <- "PASS"
} else if (has_upb_ratio) {
  cat("ERROR: upb_pct_remaining detected (H2 logic), should not be present.\n")
  status_feat <- "FAIL"
} else {
  cat("ERROR: current_upb missing.\n")
  status_feat <- "FAIL"
}
add_check("5.1", "H1 Feature Set", status_feat,
          ifelse(has_raw_upb, "Raw UPB Present", "Raw UPB Missing"),
          "H1 must use raw UPB; no ratio-based features.")

# 5B. Binary Delinquency
delq_vals <- unique(final_dt$if_delq_sts)
is_binary_delq <- all(delq_vals %in% c(0, 1, NA))
cat(sprintf("Delinquency Values found: %s\n", paste(head(delq_vals), collapse=", ")))
add_check("5.2", "Binary Delinquency", ifelse(is_binary_delq, "PASS", "FAIL"),
          "Values must be 0/1/NA for model input.")

# --- 6. Graph Construction Readiness ---------------------------------------
cat("\n--- 6. GRAPH CONSTRUCTION READINESS ---\n")

# 6A. Graph Keys
has_keys <- all(c("Geo_Key", "Lender_Key") %in% names(final_dt))
cat(ifelse(has_keys, "Geo_Key and Lender_Key are present.\n",
           "CRITICAL: Missing Geo_Key or Lender_Key!\n"))
add_check("6.1", "Graph Keys", ifelse(has_keys, "PASS", "FAIL"),
          "Required for graph construction.")

# 6B. Key Completeness
n_missing_keys <- final_dt[is.na(Geo_Key) | is.na(Lender_Key), .N]
cat(sprintf("Rows with missing keys: %d\n", n_missing_keys))
add_check("6.2", "Key Completeness", ifelse(n_missing_keys == 0, "PASS", "WARN"),
          sprintf("%d missing", n_missing_keys), "Nodes with missing keys will be isolated.")

# --- 7. Data Integrity (Missing Values & Duplicates) -----------------------
cat("\n--- 7. DATA INTEGRITY CHECKS ---\n")
# Missing Values
na_summary <- sapply(final_dt, function(x) mean(is.na(x)))
non_target_na_cols <- names(na_summary)[names(na_summary) != "Target_Y"]
max_na_col <- non_target_na_cols[which.max(na_summary[non_target_na_cols])]
max_na_pct <- 100 * max(na_summary[non_target_na_cols])
cat(sprintf("Max non-target NA%%: %.2f%% (column '%s')\n", max_na_pct, max_na_col))
add_check("7.1", "Missing Values", ifelse(max_na_pct < 20, "PASS", "WARN"),
          sprintf("%.2f%% in '%s'", max_na_pct, max_na_col),
          "High missing values may affect modeling.")

# Duplicate rows
duplicates <- final_dt[, .N, by = .(Loan_Sequence_Number, Monthly_Reporting_Period)][N > 1]
cat(sprintf("Duplicate loan-month observations: %d\n", nrow(duplicates)))
add_check("7.2", "Duplicate Rows", ifelse(nrow(duplicates) == 0, "PASS", "FAIL"),
          sprintf("%d duplicates found", nrow(duplicates)),
          "Each loan-month should appear only once.")

# --- 8. Data Type Validation -----------------------------------------------
cat("\n--- 8. DATA TYPE VALIDATION ---\n")
expected_numeric <- c(FINAL_SCALE_COLS)
all_types_ok <- TRUE
for (col in expected_numeric) {
  if (!is.numeric(final_dt[[col]])) {
    cat(sprintf("Type Mismatch: '%s' expected NUMERIC, but is %s.\n", col, class(final_dt[[col]])))
    all_types_ok <- FALSE
  }
}
if (all_types_ok) cat("All key numeric columns have correct types.\n")

# --- 9. Preprocessing Parameters Check ------------------------------------
cat("\n--- 9. PREPROCESSING PARAMETERS CHECK ---\n")

# Only check Min and Max (remove Median)
invalid_params <- loaded_data$preproc_params[is.na(Min) | is.na(Max)]

if (nrow(invalid_params) > 0) {
  cat("Some parameters contain NA values:\n")
  print(invalid_params, row.names = FALSE)
} else {
  cat("All preprocessing parameters are valid.\n")
}

add_check("9.1", "Preprocessing Params Validity",
          ifelse(nrow(invalid_params) == 0, "PASS", "FAIL"),
          sprintf("%d invalid params", nrow(invalid_params)),
          "Scaling params must be non-NA and train-only.")


# --- 10. Final Summary -----------------------------------------------------
cat("\n--- 10. FINAL SUMMARY ---\n")
print(validation_summary)
fwrite(validation_summary, file.path(SAVE_DIR, "validation_summary_h1.csv"))
cat(sprintf("\nSaved validation summary to: %s\n", file.path(SAVE_DIR, "validation_summary_h1.csv")))
