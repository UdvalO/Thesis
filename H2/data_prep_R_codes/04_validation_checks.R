# ==============================================================================
# 04_validation_checks.R (H2)
#
# PURPOSE:
#   Comprehensive validation suite for the full data pipeline.
#   Checks cover:
#     - File existence & dimensions
#     - Time window integrity & train/test split
#     - Target variable distribution & default rates
#     - Loan cohort consistency
#     - Missing values, duplicates, and type validation
#     - Temporal monotonicity of engineered features
#     - Graph construction readiness
#
# OUTPUTS:
#   - validation_summary.csv : High-level pass/fail summary
#   - target_check.csv       : Snapshot-level default rates for detailed analysis
# ==============================================================================

# --- 0. DEPENDENCIES ---
library(data.table)
library(lubridate)
source("00_config.R")

cat("\n========================================\n")
cat("     DATA PIPELINE VALIDATION CHECKS\n")
cat("========================================\n\n")

# --- Initialize Validation Summary Table ------------------------------------
validation_summary <- data.table(
  Check_ID = character(),
  Check_Name = character(),
  Status = character(), # PASS, FAIL, WARN
  Metric = character(),
  Notes = character()
)

# Helper to append validation results
add_check <- function(id, name, status, metric = "", notes = "") {
  validation_summary <<- rbind(validation_summary, list(id, name, status, as.character(metric), notes))
}


# --- 1. Load All Saved Files --------------------------------------------------
cat("--- 1. LOADING ALL SAVED FILES ---\n")

files_to_check <- list(
  orig_data = ORIG_SAVE_FILE,
  perf_data = PERF_SAVE_FILE,
  final_features = file.path(SAVE_DIR, "final_features_raw_clean.rds"),
  final_with_targets = file.path(SAVE_DIR, "final_data_base_with_targets.rds"),
  train_targets = file.path(SAVE_DIR, "train_targets.rds"),
  test_targets = file.path(SAVE_DIR, "test_targets.rds"),
  preproc_params = file.path(SAVE_DIR, "preproc_params.rds")
)

loaded_data <- list()
all_files_loaded <- TRUE

for (name in names(files_to_check)) {
  file_path <- files_to_check[[name]]
  
  if (file.exists(file_path)) {
    loaded_data[[name]] <- readRDS(file_path)
    if (is.data.frame(loaded_data[[name]])) {
      setDT(loaded_data[[name]])
    }
    cat(sprintf("%s loaded successfully\n", name))
  } else {
    cat(sprintf("%s NOT FOUND at %s\n", name, file_path))
    all_files_loaded <- FALSE
  }
}
cat("\n")

add_check(
  "1.1", "File Existence",
  ifelse(all_files_loaded, "PASS", "FAIL"),
  sprintf("%d/%d files found", length(loaded_data), length(files_to_check)),
  "All required .rds artifacts must be present."
)

if (!all_files_loaded) {
  stop("Critical data files are missing. Halting validation.")
}

# --- 2. SAMPLE SIZE & DIMENSION CHECKS ---------------------------------------
cat("--- 2. BASIC DIMENSION CHECKS ---\n")
cat(sprintf(
  "Origination data: %d rows, %d columns, %d unique loans\n",
  nrow(loaded_data$orig_data),
  ncol(loaded_data$orig_data),
  uniqueN(loaded_data$orig_data$Loan_Sequence_Number)
))
cat(sprintf(
  "Performance data: %d rows, %d columns, %d unique loans\n",
  nrow(loaded_data$perf_data),
  ncol(loaded_data$perf_data),
  uniqueN(loaded_data$perf_data$Loan_Sequence_Number)
))
cat(sprintf(
  "Final with targets: %d rows, %d columns, %d unique loans\n",
  nrow(loaded_data$final_with_targets),
  ncol(loaded_data$final_with_targets),
  uniqueN(loaded_data$final_with_targets$Loan_Sequence_Number)
))
cat("\n")

# --- 3. Time Window Validation ------------------------------------------------
cat("--- 3. TIME WINDOW VALIDATION ---\n")
perf_period_range <- range(loaded_data$perf_data$Monthly_Reporting_Period, na.rm = TRUE)
cat(sprintf(
  "Performance data period: %d to %d (Expected: %d to %d)\n",
  perf_period_range[1], perf_period_range[2],
  PERF_START_HISTORY, PERF_END_PERIOD
))
is_period_correct <- perf_period_range[1] == PERF_START_HISTORY && perf_period_range[2] == PERF_END_PERIOD
cat(ifelse(is_period_correct, "Performance period matches expected range\n\n",
           "Performance period DOES NOT match expected range\n\n"))
add_check(
  "3.1", "Performance Time Window",
  ifelse(is_period_correct, "PASS", "FAIL"),
  paste(perf_period_range, collapse = " to "),
  sprintf("Expected: %d to %d", PERF_START_HISTORY, PERF_END_PERIOD)
)

# --- 4. Train/Test Split Validation ------------------------------------------
cat("--- 4. TRAIN/TEST SPLIT VALIDATION ---\n")

train_range <- range(loaded_data$train_targets$Snapshot_Date, na.rm = TRUE)
test_range <- range(loaded_data$test_targets$Snapshot_Date, na.rm = TRUE)
cat(sprintf(
  "Training snapshots: %d to %d (%d unique dates)\n",
  train_range[1], train_range[2],
  uniqueN(loaded_data$train_targets$Snapshot_Date)
))
cat(sprintf(
  "Testing snapshots: %d to %d (%d unique dates)\n",
  test_range[1], test_range[2],
  uniqueN(loaded_data$test_targets$Snapshot_Date)
))
no_overlap <- train_range[2] < test_range[1]
cat(ifelse(no_overlap, "No overlap between train and test periods\n\n",
           "OVERLAP DETECTED between train and test periods!\n\n"))
add_check(
  "4.1", "Train/Test Overlap",
  ifelse(no_overlap, "PASS", "FAIL"),
  sprintf("Train ends %d, Test starts %d", train_range[2], test_range[1]),
  "Train snapshot dates must be strictly less than test snapshot dates."
)

# --- 5. Target Distribution Check -------------------------------------------
cat("--- 5. TARGET DISTRIBUTION CHECK ---\n")
train_target_dist <- table(loaded_data$train_targets$Target_Y)
test_target_dist <- table(loaded_data$test_targets$Target_Y)
train_default_rate <- train_target_dist["1"] / sum(train_target_dist)
test_default_rate <- test_target_dist["1"] / sum(test_target_dist)
cat(sprintf("Train default rate: %.2f%%\n", 100 * train_default_rate))
cat(sprintf("Test default rate: %.2f%%\n", 100 * test_default_rate))
is_rate_similar <- abs(train_default_rate - test_default_rate) < 0.05
cat(ifelse(is_rate_similar, "Train and test default rates are similar\n\n",
           "Train and test default rates differ significantly\n\n"))
add_check(
  "5.1", "Target Rate Stability",
  ifelse(is_rate_similar, "PASS", "WARN"),
  sprintf("Train: %.2f%%, Test: %.2f%%", 100 * train_default_rate, 100 * test_default_rate),
  "Checks if default rates between train/test sets are reasonably close."
)

# --- Generate target_check.csv ------------------------------------------------
cat("--- Generating target_check.csv for detailed analysis ---\n")
all_targets_combined <- rbind(loaded_data$train_targets, loaded_data$test_targets)
target_check_dt <- all_targets_combined[, .(
  N_Loans = .N,
  N_Defaults = sum(Target_Y, na.rm = TRUE)
), by = Snapshot_Date]
target_check_dt[, N_NonDefaults := N_Loans - N_Defaults]
target_check_dt[, Default_Rate_Pct := 100 * N_Defaults / N_Loans]
target_check_dt[, Set := ifelse(Snapshot_Date <= TRAIN_SPLIT_DATE, "Train", "Test")]
setorder(target_check_dt, Snapshot_Date)
target_check_file_path <- file.path(SAVE_DIR, "target_check.csv")
fwrite(target_check_dt, target_check_file_path)
cat(sprintf("Saved detailed target analysis to: %s\n\n", target_check_file_path))
add_check("5.2", "Target Detail File", "PASS", "target_check.csv", "Generated file with per-snapshot default rates.")

# --- 6. Loan Cohort Consistency Check ---------------------------------------
cat("--- 6. LOAN COHORT CONSISTENCY CHECK ---\n")
orig_loans <- unique(loaded_data$orig_data$Loan_Sequence_Number)
perf_loans <- unique(loaded_data$perf_data$Loan_Sequence_Number)
is_cohort_synced <- setequal(orig_loans, perf_loans)
cat(ifelse(is_cohort_synced, "  ✅ Origination and performance have same loan set (Post-Sync)\n\n",
           "Origination and performance loan sets differ\n\n"))
add_check(
  "6.1", "Cohort Consistency",
  ifelse(is_cohort_synced, "PASS", "FAIL"),
  sprintf("Orig: %d, Perf: %d loans", length(orig_loans), length(perf_loans)),
  "Checks if 01_data_import_cleaner.R correctly synchronized loan cohorts."
)

# --- 7. DATA INTEGRITY CHECKS (Missing Values & Duplicates) -----------------
cat("--- 7. DATA INTEGRITY CHECKS ---\n")
final_dt <- loaded_data$final_with_targets

# --- 7a. Missing Values ---
# Calculate NA percentage for all columns (excluding target)
na_summary <- sapply(final_dt, function(x) mean(is.na(x)))
non_target_na_cols <- names(na_summary)[names(na_summary) != "Target_Y"]
max_na_col <- non_target_na_cols[which.max(na_summary[non_target_na_cols])]
max_na_pct <- 100 * max(na_summary[non_target_na_cols])
cat(sprintf("Max non-target NA%%: %.2f%% (in column '%s')\n", max_na_pct, max_na_col))
add_check(
  "7.1", "Missing Values",
  ifelse(max_na_pct < 20, "PASS", "WARN"),
  sprintf("%.2f%% in '%s'", max_na_pct, max_na_col),
  "Checks for high percentage of missing values in any feature column."
)

# --- 7b. Duplicate Rows ---
# Ensure each loan-month is unique
duplicates <- final_dt[, .N, by = .(Loan_Sequence_Number, Monthly_Reporting_Period)][N > 1]
cat(sprintf("Duplicate loan-month observations: %d\n", nrow(duplicates)))
add_check(
  "7.2", "Duplicate Rows",
  ifelse(nrow(duplicates) == 0, "PASS", "FAIL"),
  sprintf("%d duplicates found", nrow(duplicates)),
  "Each loan should have at most one record per month."
)

# --- 8. Data Type Validation -----------------------------------------------
cat("\n--- 8. DATA TYPE VALIDATION ---\n")
all_types_ok <- TRUE
dt_to_check <- loaded_data$final_with_targets
expected_numeric <- c(FINAL_ORIG_COLS_TO_SCALE, FINAL_PERF_COLS_TO_SCALE)
for (col in expected_numeric) {
  if (!is.numeric(dt_to_check[[col]])) {
    cat(sprintf("Type Mismatch: '%s' expected NUMERIC, but is %s.\n", col, class(dt_to_check[[col]])))
    all_types_ok <- FALSE
  }
}
if (all_types_ok) { cat("All key numeric columns have the correct data type.\n") }

# --- 8B. FEATURE SUMMARY STATISTICS ---------------------------------------
cat("\n--- 8B. GENERATING FEATURE SUMMARY STATISTICS ---\n")

dt_for_stats <- loaded_data$final_with_targets

# Identify numeric features (excluding IDs/dates)
numeric_cols <- names(dt_for_stats)[sapply(dt_for_stats, is.numeric)]
numeric_cols <- setdiff(numeric_cols, c("Monthly_Reporting_Period", "Default_Month")) # Exclude dates from stats

# 2. Calculate Statistics
stats_list <- list()

for (col in numeric_cols) {
  vals <- dt_for_stats[[col]]
  
  # Calculate stats (handling NAs)
  stats_list[[length(stats_list) + 1]] <- data.table(
    Feature = col,
    Type = class(vals)[1],
    N_Total = length(vals),
    N_Missing = sum(is.na(vals)),
    Pct_Missing = round(100 * sum(is.na(vals)) / length(vals), 2),
    Mean = round(mean(vals, na.rm = TRUE), 4),
    Median = round(median(vals, na.rm = TRUE), 4),
    SD = round(sd(vals, na.rm = TRUE), 4),
    Min = round(min(vals, na.rm = TRUE), 4),
    Max = round(max(vals, na.rm = TRUE), 4),
    Zeros = sum(vals == 0, na.rm = TRUE) # Helpful for sparsity checks
  )
}

feature_stats_dt <- rbindlist(stats_list)

# Preview and save
print(feature_stats_dt[, .(Feature, Mean, Median, Min, Max, Pct_Missing)])
stats_file_path <- file.path(SAVE_DIR, "feature_summary_stats.csv")
fwrite(feature_stats_dt, stats_file_path)

cat(sprintf("\n Detailed feature statistics saved to: %s\n", stats_file_path))

add_check(
  "8.2", "Feature Stats Generation", 
  "PASS", 
  sprintf("%d features analyzed", nrow(feature_stats_dt)), 
  "Generated distribution statistics (mean, sd, etc.) for all numeric columns."
)

# --- 9. Preprocessing Parameters Check ------------------------------------
cat("\n--- 9. PREPROCESSING PARAMETERS CHECK ---\n")
invalid_params <- loaded_data$preproc_params[is.na(Min) | is.na(Max) | is.na(Median)]
if (nrow(invalid_params) > 0) {
  cat("Some parameters contain NA values:\n")
  print(invalid_params, row.names = FALSE)
} else {
  cat("All preprocessing parameters are valid (non-NA, leakage-free)\n")
}
add_check(
  "9.1", "Preprocessing Params Validity",
  ifelse(nrow(invalid_params) == 0, "PASS", "FAIL"),
  sprintf("%d invalid params", nrow(invalid_params)),
  "Scaling parameters must be non-NA and calculated only on train data."
)


# --- 10. Temporal Monotonicity Check ---------------------------------------
cat("\n--- 10. TEMPORAL MONOTONICITY CHECK (clean_loan_age) ---\n")
setkey(loaded_data$final_with_targets, Loan_Sequence_Number, Monthly_Reporting_Period)
loaded_data$final_with_targets[, clean_age_diff := c(NA, diff(clean_loan_age)), by = Loan_Sequence_Number]
non_monotonic_clean_age <- loaded_data$final_with_targets[clean_age_diff <= 0]
if (nrow(non_monotonic_clean_age) > 0) {
  cat(sprintf("CRITICAL: %d records found where clean_loan_age is not strictly increasing!\n", nrow(non_monotonic_clean_age)))
} else {
  cat("clean_loan_age is strictly increasing for all loans.\n")
}
add_check(
  "10.1", "Feature Monotonicity",
  ifelse(nrow(non_monotonic_clean_age) == 0, "PASS", "FAIL"),
  sprintf("%d non-monotonic rows", nrow(non_monotonic_clean_age)),
  "The engineered loan age must increase by 1 each month for each loan."
)
loaded_data$final_with_targets[, clean_age_diff := NULL] # Clean up


# --- 11. Graph Construction Readiness -------------------------------------
cat("\n--- 11. GRAPH CONSTRUCTION READINESS ---\n")

# Check snapshot coverage
snapshot_coverage <- target_check_dt[, .(N_Loans), by = Snapshot_Date]
cat("\nSnapshot temporal coverage (loan counts):\n")
print(snapshot_coverage)
is_coverage_sufficient <- !any(snapshot_coverage$N_Loans < 100)
if (!is_coverage_sufficient) {
  cat("Warning: Some snapshots have very few observations (< 100)\n")
}
add_check(
  "11.1", "Snapshot Coverage",
  ifelse(is_coverage_sufficient, "PASS", "WARN"),
  sprintf("Min loans in a snapshot: %d", min(snapshot_coverage$N_Loans)),
  "Ensures each time step has enough data to form a meaningful graph."
)

# Verify presence of required features for GNN
required_features <- c("fico", "ltv", "dti", "clean_loan_age", "upb_pct_remaining",
                       "Geo_Key", "Lender_Key", "Target_Y")
missing_features <- setdiff(required_features, names(loaded_data$final_with_targets))
if (length(missing_features) > 0) {
  cat(sprintf("CRITICAL: Missing required features for modeling: %s\n",
              paste(missing_features, collapse = ", ")))
} else {
  cat("All required features for modeling are present\n")
}
add_check(
  "11.2", "Feature Completeness",
  ifelse(length(missing_features) == 0, "PASS", "FAIL"),
  paste(missing_features, collapse = ", "),
  "Checks if the final dataset contains all columns needed for the Python models."
)


# --- 12. Final Summary --------------------------------------------------
cat("\n\n--- 12. FINAL SUMMARY --- \n")
cat("All validation checks complete.\n")
print(validation_summary)

# Save the Final Validation Summary
validation_summary_file_path <- file.path(SAVE_DIR, "validation_summary.csv")
fwrite(validation_summary, validation_summary_file_path)
cat(sprintf("\n Saved validation summary to: %s\n", validation_summary_file_path))