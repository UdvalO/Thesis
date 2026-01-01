# ==============================================================================
# 001_data_import_cleaner.R (H1)
#
# PURPOSE:
#   Clean and synchronize raw origination and performance .txt files for analysis.
#
# STEPS:
#   1. Read raw files efficiently by year/quarter.
#   2. Drop loans with any NA delinquency status.
#   3. Apply initial filters (first-time homebuyer, FICO/LTV/DTI validity).
#   4. Apply Default Timer filter to exclude early defaults.
#   5. Filter performance data to defined time window.
#   6. Synchronize origination and performance datasets.
#   7. Convert numeric columns and save as RDS.
#   8. Perform validation checks (record counts, NA checks).
#
# INPUTS:
#   - historical_data_YYYYQ#.txt (Raw Origination Files)
#   - historical_data_time_YYYYQ#.txt (Raw Performance Files)
#   - 000_config.R
#
# OUTPUTS:
#   - orig_data_cleaned.rds
#   - perf_data_cleaned.rds
# ==============================================================================


# --- 0. DEPENDENCIES ---
source("000_config.R")
library(lubridate)
library(data.table)

# --- 1. EFFICIENT READ FUNCTION ---
# Efficiently reads 2009-2010 vintages using column maps from 000_config.R.
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
  
  # --- A. READ RAW ORIGINATION & PERFORMANCE DATA ---------------------------
  orig_data <- read_and_clean_efficient("historical_data_", "historical_data_", orig_cols_map)
  perf_data <- read_and_clean_efficient("historical_data_", "historical_data_time_", perf_cols_map)
  
  # --- B. LOAN-LEVEL NA DELINQUENCY FILTER ----------------------------------
  # Identify loans with at least one NA delinquency status and
  # remove them entirely from both datasets to maintain consistency
  cat("\n--- Identifying and dropping loans with NA delinquency ---\n")
  
  # 1. Get the list of "bad" loan IDs from perf_data
  loans_with_na_delq <- unique(perf_data[is.na(if_delq_sts), Loan_Sequence_Number])
  
  if (length(loans_with_na_delq) > 0) {
    cat(sprintf("Found %d unique loans with at least one NA in 'if_delq_sts'.\n", length(loans_with_na_delq)))
    
    # 2. Get row counts before dropping
    orig_rows_before <- nrow(orig_data)
    perf_rows_before <- nrow(perf_data)
    
    # 3. Filter both datasets to remove these loans entirely
    orig_data <- orig_data[!(Loan_Sequence_Number %in% loans_with_na_delq)]
    perf_data <- perf_data[!(Loan_Sequence_Number %in% loans_with_na_delq)]
    
    cat(sprintf("Dropped %d rows from orig_data.\n", orig_rows_before - nrow(orig_data)))
    cat(sprintf("Dropped %d rows from perf_data.\n", perf_rows_before - nrow(perf_data)))
    cat("Both datasets are now synchronized and clean of NA delinquency.\n")
    
  } else {
    cat("No loans with NA delinquency status found.\n")
  }
  
  # --- C. DATA TYPE CONVERSION & BASIC CLEANING -----------------------------
  cat("\n--- Performing Data Type Conversion and Initial Cleaning ---\n")
  
  # Convert numeric columns
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
  rows_before_validity <- nrow(orig_data)
  
  # Filtering based on the range specificed in FM User Guide
  orig_data <- orig_data[
    fico >= 300 & fico <= 850 &
      ltv > 0 & ltv <= 100 &
      dti > 0 & dti <= 65
  ]
  
  cat(sprintf("Removed %d rows with invalid FICO/LTV/DTI values.\n",
              rows_before_validity - nrow(orig_data)))
  
  
  # --- E. DEFAULT TIMING (d_timer) FILTER -----------------------------------
  # Exclude loans that defaulted before the observation window
  cat("\n--- STAGE 2: Applying d_timer filter ---\n")
  
  REFERENCE_DATE <- ymd("2009-02-01") # the Freddie Mac dataset start Feb, 2009
  
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
  
  rows_before_dtimer <- nrow(orig_data)
  orig_data <- orig_data[d_timer >= 35] # filtering out defaults before Jan 2012
  
  cat(sprintf("Dropped %d loans that defaulted before Jan 2012.\n",
              rows_before_dtimer - nrow(orig_data)))
  
  # Remove helper column and resync performance data
  orig_data[, d_timer := NULL]
  perf_data <- perf_data[Loan_Sequence_Number %in% orig_data$Loan_Sequence_Number]
  
  cat(sprintf("Final dataset contains %d unique loans for analysis.\n",
              uniqueN(orig_data$Loan_Sequence_Number)))
  
  # --- F. PERFORMANCE TIME WINDOW FILTER ------------------------------------
  cat(sprintf("\n--- Filtering performance data (%d to %d) ---\n", PERF_START_HISTORY, PERF_END_PERIOD))
  perf_data <- perf_data[Monthly_Reporting_Period >= PERF_START_HISTORY &
                           Monthly_Reporting_Period <= PERF_END_PERIOD]
  
  # --- G. FINAL DATASET SYNCHRONIZATION -------------------------------------
  cat("\n--- Final Synchronization of Origination and Performance Cohorts ---\n")
  
  common_loan_ids <- intersect(
    unique(orig_data$Loan_Sequence_Number),
    unique(perf_data$Loan_Sequence_Number)
  )
  
  cat(sprintf("Loans in orig_data (post-filter): %d\n", uniqueN(orig_data$Loan_Sequence_Number)))
  cat(sprintf("Loans in perf_data (post-filter): %d\n", uniqueN(perf_data$Loan_Sequence_Number)))
  cat(sprintf("Common loans (intersection): %d\n", length(common_loan_ids)))
  
  orig_data <- orig_data[Loan_Sequence_Number %in% common_loan_ids]
  perf_data <- perf_data[Loan_Sequence_Number %in% common_loan_ids]
  
  cat(sprintf("\nFinal synchronized dataset: %d unique loans\n", 
              uniqueN(orig_data$Loan_Sequence_Number)))
  
  if (setequal(unique(orig_data$Loan_Sequence_Number), 
               unique(perf_data$Loan_Sequence_Number))) {
    cat("Origination and performance loan sets are now perfectly aligned.\n")
  } else {
    cat("ERROR: Loan sets still don't match after synchronization!\n")
  }
  
  # --- H. FINAL TYPE CONVERSION & SAVE --------------------------------------
  perf_data[, Monthly_Reporting_Period := as.integer(Monthly_Reporting_Period)]
  for (col in PERF_NUMERIC_COLS) {
    perf_data[, (col) := as.numeric(get(col))]
  }
  
  # --- I. SAVE PROCESSED DATA  --------------------------------------
  cat("\nData processing complete. Saving cleaned data to RDS.\n")
  saveRDS(orig_data, ORIG_SAVE_FILE)
  saveRDS(perf_data, PERF_SAVE_FILE)
}

# --- 3. VALIDATION CHECK ---
# Sanity checks to confirm data integrity after all filters and synchronization
cat("\n--- VALIDATION (Post-Processing Filter Check) ---\n")

if (exists("orig_data") && exists("perf_data") && nrow(orig_data) > 0 && nrow(perf_data) > 0) {
  
  # --- A. BASIC DATASET SUMMARY --------------------------------------------
  cat(sprintf("Origination records: %d\n", nrow(orig_data)))
  cat(sprintf("Performance period range: %d to %d\n",
              min(perf_data$Monthly_Reporting_Period, na.rm=TRUE),
              max(perf_data$Monthly_Reporting_Period, na.rm=TRUE)))
  cat(sprintf("Unique loans in origination data: %d\n", uniqueN(orig_data$Loan_Sequence_Number)))
  cat(sprintf("Unique loans in filtered performance data: %d\n", uniqueN(perf_data$Loan_Sequence_Number)))
  
  # --- B. NA VALIDATION -----------------------------------------------------
  # Confirm that key numeric columns contain no unexpected missing values
  cat("\n--- NA (Null) Validation Check ---\n")
  
  na_summary_orig <- sapply(orig_data[, .SD, .SDcols = ORIG_NUMERIC_COLS], function(x) sum(is.na(x)))
  print(na_summary_orig)
  
  na_summary_perf <- sapply(perf_data[, .SD, .SDcols = PERF_NUMERIC_COLS], function(x) sum(is.na(x)))
  print(na_summary_perf)
  
} else {
  # Fail-fast warning if filtering removed all data
  cat("Warning: Dataframes are empty after filtering. Check filters and file paths.\n")
}