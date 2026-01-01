# ==============================================================================
# 01_data_import_cleaner.R (H2)
#
# PURPOSE:
#   Import, clean, and synchronize raw Freddie Mac origination and performance
#   data for the H1 (strict replication) pipeline.
#
# MAIN STEPS:
#   1. Efficiently read raw .txt files by vintage year and quarter.
#   2. Drop loans with any missing delinquency status.
#   3. Convert data types and binary flags.
#   4. Apply validity filters (FICO, LTV, DTI).
#   5. Apply default-timer (d_timer) filter to remove early defaults.
#   6. Restrict performance data to the defined historical window.
#   7. Synchronize origination and performance datasets.
#   8. Save cleaned datasets as RDS files.
#
# INPUTS:
#   - Raw Freddie Mac .txt files
#   - Column maps and constants from 00_config.R
#
# OUTPUTS:
#   - ORIG_SAVE_FILE
#   - PERF_SAVE_FILE
# ==============================================================================

# --- 0. DEPENDENCIES ---
source("00_config.R")
library(lubridate)
library(data.table)

# --- 1. EFFICIENT READ FUNCTION ---
# Efficiently reads 2009-2010 vintages using column maps from 00_config.R.
read_and_clean_efficient <- function(folder_type_base, file_type_base, col_map) {
  all_data <- list()
  col_indices <- as.vector(col_map)
  
  for (year in VINTAGE_YEARS) {
    for (q in 1:4) {
      quarter_string <- paste0(year, "Q", q)
      file_path <- file.path(BASE_DIR, paste0(folder_type_base, quarter_string), paste0(file_type_base, quarter_string, ".txt"))
      
      if (!file.exists(file_path)) next
      cat(sprintf("Reading: %s\n", file_path))
      
      dt <- fread(file_path, sep = "|", header = FALSE,
                  select = col_indices,
                  na.strings = NA_STRINGS,
                  showProgress = FALSE)
      
      setnames(dt, old = names(dt), new = names(col_map))
      all_data[[length(all_data) + 1]] <- dt
    }
  }
  return(rbindlist(all_data, use.names = TRUE, fill = TRUE))
}

# --- 2. CONDITIONAL EXECUTION ---
# To save time, if files exist no need to run again
if (file.exists(ORIG_SAVE_FILE) && file.exists(PERF_SAVE_FILE)) {
  cat("--- Loading pre-processed data from RDS files ---\n")
  orig_data <- readRDS(ORIG_SAVE_FILE)
  perf_data <- readRDS(PERF_SAVE_FILE)
  
} else {
  cat("--- Reading and processing raw data ---\n")
  
  # --- A. READ RAW DATA -----------------------------------------------------
  orig_data <- read_and_clean_efficient("historical_data_", "historical_data_", orig_cols_map)
  perf_data <- read_and_clean_efficient("historical_data_", "historical_data_time_", perf_cols_map)
  
  # --- B. DROP LOANS WITH NA DELINQUENCY ------------------------------------
  # Any loan with at least one NA delinquency record is removed entirely.
  cat("\n--- Identifying and dropping loans with NA delinquency ---\n")
  
  loans_with_na_delq <- unique(perf_data[is.na(if_delq_sts), Loan_Sequence_Number])
  
  if (length(loans_with_na_delq) > 0) {
    cat(sprintf("  Found %d unique loans with at least one NA in 'if_delq_sts'.\n", length(loans_with_na_delq)))
    orig_data <- orig_data[!(Loan_Sequence_Number %in% loans_with_na_delq)]
    perf_data <- perf_data[!(Loan_Sequence_Number %in% loans_with_na_delq)]
    cat("  ✅ Dropped loans with NA delinquency.\n")
  } 
  
  # --- C. DATA TYPE CONVERSION & BINARY FLAGS -------------------------------
  cat("\n--- Performing Data Type Conversion and Initial Cleaning ---\n")
  
  # Convert numeric origination fields
  orig_data[, (ORIG_NUMERIC_COLS) := lapply(.SD, as.numeric), .SDcols = ORIG_NUMERIC_COLS]
  
  # Encode categorical flags as binary indicators
  orig_data[, `:=`(
    if_fthb = fcase(if_fthb == "Y", 1L, if_fthb == "N", 0L, default = NA_integer_),
    if_prim_res = fcase(if_prim_res == "P", 1L, default = 0L),
    if_corr = fcase(if_corr == "C", 1L, default = 0L),
    if_sf = fcase(if_sf == "SF", 1L, default = 0L),
    if_purc = fcase(if_purc == "P", 1L, default = 0L),
    if_sc = fcase(if_sc == "Y", 1L, default = 0L)
  )]
  
  # Remove loans with unknown first-time homebuyer status
  rows_before_fthb_filter <- nrow(orig_data)
  orig_data <- orig_data[!is.na(if_fthb)]
  cat(sprintf("Dropped %d rows from orig_data due to unknown first-time homebuyer status.\n",
              rows_before_fthb_filter - nrow(orig_data)))
  
  # --- D. VALIDITY FILTERS (FICO / LTV / DTI) -------------------------------
  cat("\n--- Applying validity filters for FICO / LTV / DTI ---\n")
  
  # Filtering based on the range specificed in FM User Guide
  orig_data <- orig_data[
    fico >= 300 & fico <= 850 &
      ltv > 0 & ltv <= 100 &
      dti > 0 & dti <= 65
  ]
  
  # --- E. DEFAULT TIMING (d_timer) FILTER -----------------------------------
  # Exclude loans that defaulted before the observation window
  cat("\n--- STAGE 2: Applying d_timer filter ---\n")
  REFERENCE_DATE <- ymd("2009-02-01")
  
  # Treating RA as default (explanation in Thesis section 3.1)
  perf_data[, delq_numeric := fcase(
    if_delq_sts == "RA", 3L,
    default = suppressWarnings(as.integer(if_delq_sts))
  )]
  
  # Identify first default date per loan
  default_dates <- perf_data[!is.na(delq_numeric) & delq_numeric >= 3,
                             .(default_period = min(Monthly_Reporting_Period)),
                             by = Loan_Sequence_Number]
  
  default_dates[, default_date := ymd(paste0(default_period, "01"))]
  default_dates[, d_timer := as.period(interval(REFERENCE_DATE, default_date)) %/% months(1)]
  
  # Attach d_timer to origination data
  orig_data[default_dates, d_timer := i.d_timer, on = "Loan_Sequence_Number"]
  orig_data[is.na(d_timer), d_timer := 1000L] ## assigning high value to keep the healthy loans in the dataset
  
  orig_data <- orig_data[d_timer >= 35] # filtering out defaults before Jan 2012
  # Remove helper column and resync performance data
  orig_data[, d_timer := NULL]
  
  # Sync Perf Data
  perf_data <- perf_data[Loan_Sequence_Number %in% orig_data$Loan_Sequence_Number]
  
  # --- F. PERFORMANCE TIME WINDOW FILTER ------------------------------------
  perf_data <- perf_data[Monthly_Reporting_Period >= PERF_START_HISTORY &
                           Monthly_Reporting_Period <= PERF_END_PERIOD]
  
  # --- G. FINAL DATASET SYNCHRONIZATION -------------------------------------
  common_ids <- intersect(orig_data$Loan_Sequence_Number, perf_data$Loan_Sequence_Number)
  orig_data <- orig_data[Loan_Sequence_Number %in% common_ids]
  perf_data <- perf_data[Loan_Sequence_Number %in% common_ids]
  
  # --- H. FINAL TYPE CONVERSION & SAVE --------------------------------------
  perf_data[, Monthly_Reporting_Period := as.integer(Monthly_Reporting_Period)]
  for (col in PERF_NUMERIC_COLS) {
    perf_data[, (col) := as.numeric(get(col))]
  }
  
  cat("\nData processing complete. Saving cleaned data to RDS.\n")
  saveRDS(orig_data, ORIG_SAVE_FILE)
  saveRDS(perf_data, PERF_SAVE_FILE)
}