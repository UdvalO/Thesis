# ==============================================================================
# 02_data_prep_feature.R (H2)
#
# PURPOSE:
#   Construct a balanced modeling dataset by sampling loans, cleaning features,
#   engineering defaults, and preparing inputs for downstream modeling.
#
# MAIN STEPS:
#   1. Load cleaned origination and performance datasets.
#   2. Identify relevant defaulting loans within the study window.
#   3. Sample ~250k loans with a 5% default rate (stratified negatives).
#   4. Cap outliers and impute missing numeric values.
#   5. Clean categorical and binary features.
#   6. Engineer loan-age and default timing features.
#   7. Merge origination and performance features into a final dataset.
#
# INPUTS:
#   - ORIG_SAVE_FILE
#   - PERF_SAVE_FILE
#
# OUTPUTS:
#   - final_features_raw_clean.rds
# ==============================================================================

# --- 0. DEPENDENCIES ---
source("00_config.R")
library(data.table)
library(tidyverse)
library(magrittr)
set.seed(42) # for reproducibility

# --- HELPER FUNCTIONS -------------------------------------------------------
# Performs leakage-safe outlier capping and median imputation

# Compute statistical mode (used for categorical imputation)
get_mode <- function(v) {
  v_no_na <- v[!is.na(v)]
  if (length(v_no_na) == 0) return(NA)
  uv <- unique(v_no_na)
  uv[which.max(tabulate(match(v_no_na, uv)))]
}

# Cap numeric outliers at 1st/99th percentiles and median-impute missing values
perform_capping_and_imputation <- function(DT, numeric_cols) {
  valid_cols <- intersect(numeric_cols, names(DT))
  
  for (col in valid_cols) {
    # Ensure numeric type
    if (!is.numeric(DT[[col]])) DT[, (col) := suppressWarnings(as.numeric(get(col)))]
    
    # Outlier capping
    tryCatch({
      qs <- quantile(DT[[col]], probs = c(0.01, 0.99), na.rm = TRUE)
      DT[get(col) < qs[1], (col) := qs[1]]
      DT[get(col) > qs[2], (col) := qs[2]]
    }, error = function(e) {})
    
    # Median imputation
    med <- median(DT[[col]], na.rm = TRUE)
    if (!is.na(med)) DT[is.na(get(col)), (col) := med]
  }
  return(DT)
}

# --- 2. LOAD PERFORMANCE DATA -----------------------------------------------
cat("--- 1. Loading Data ---\n")

perf_data <- readRDS(PERF_SAVE_FILE)
setDT(perf_data)

# --- 3. IDENTIFY RELEVANT DEFAULTERS -----------------------------------------
# A loan is considered a defaulter if it ever reaches 90+ days delinquent.
default_info <- perf_data[!is.na(delq_numeric) & delq_numeric >= 3,
                          .(Default_Month = min(Monthly_Reporting_Period)), by = "Loan_Sequence_Number"]

relevant_defaulters <- default_info[Default_Month >= START_PERIOD_TRAIN & Default_Month <= PERF_END_PERIOD, Loan_Sequence_Number]

# --- 4. LOAD ORIGINATION DATA -----------------------------------------------
orig_data <- readRDS(ORIG_SAVE_FILE)
setDT(orig_data)

# --- 5. SAMPLING STRATEGY ----------------------------------------------------
# Target: 250,000 loans with ~5% default rate
TARGET_TOTAL <- 250000; 
TARGET_POS_RATE <- 0.05 # 5% as observed in the Reference Model 
TARGET_POS_COUNT <- floor(TARGET_TOTAL * TARGET_POS_RATE)
TARGET_NEG_COUNT <- TARGET_TOTAL - TARGET_POS_COUNT

# Sample defaulters
if (length(relevant_defaulters) <= TARGET_POS_COUNT) {
  sampled_pos_ids <- relevant_defaulters
} else {
  sampled_pos_ids <- sample(relevant_defaulters, TARGET_POS_COUNT)
}

# Stratified sampling for non-defaulters (Geo × Lender)
pool_neg_ids <- setdiff(unique(orig_data$Loan_Sequence_Number), relevant_defaulters)
orig_data[, Geo_Key := substr(as.character(Postal_Code), 1, 2)]
orig_data[, Lender_Key := Seller_Name]

sampling_frame <- orig_data[Loan_Sequence_Number %in% pool_neg_ids, .(Loan_Sequence_Number, Geo_Key, Lender_Key)]
sampling_frame[, strata := paste(Geo_Key, Lender_Key, sep = "_")]

strata_counts <- sampling_frame[, .N, by = strata]
strata_counts[, target_n := pmax(1, round(N / sum(N) * TARGET_NEG_COUNT))]

sampled_neg_ids <- sampling_frame[, .SD[sample(.N, min(.N, strata_counts[strata == .BY$strata, target_n]))], by = strata]$Loan_Sequence_Number

# Top-up if stratified sampling underfills target
shortfall <- TARGET_NEG_COUNT - length(sampled_neg_ids)
if (shortfall > 0) sampled_neg_ids <- c(sampled_neg_ids, sample(setdiff(pool_neg_ids, sampled_neg_ids), shortfall))

# Final sample
final_sample_ids <- c(sampled_pos_ids, sampled_neg_ids)
orig_data <- orig_data[Loan_Sequence_Number %in% final_sample_ids]
perf_data <- perf_data[Loan_Sequence_Number %in% final_sample_ids]

# --- 6. DEFAULT LOGIC & CLEANING ---------------------------------------------
# Assign first default month; non-defaulters receive sentinel value
sample_defaults <- perf_data[!is.na(delq_numeric) & delq_numeric >= 3, .(Default_Month = min(Monthly_Reporting_Period)), by = "Loan_Sequence_Number"]
perf_data[sample_defaults, Default_Month := i.Default_Month, on = "Loan_Sequence_Number"]
perf_data[is.na(Default_Month), Default_Month := 999999L] # placeholder to avoid using NA


# Cap outliers and impute missing numeric values
orig_data <- perform_capping_and_imputation(orig_data, ORIG_NUMERIC_COLS)
perf_data <- perform_capping_and_imputation(perf_data, PERF_NUMERIC_COLS)

# --- 7. FEATURE ENGINEERING --------------------------------------------------
# Impute missing categorical values using mode
char_cols <- setdiff(names(orig_data)[sapply(orig_data, is.character)], c("Loan_Sequence_Number", "Geo_Key", "Lender_Key"))
for (col in char_cols) {
  val <- get_mode(orig_data[[col]])
  if (!is.na(val)) orig_data[is.na(get(col)), (col) := val]
}

# Ensure binary flags are complete
binary_flags <- intersect(c("if_fthb", "if_prim_res", "if_corr", "if_sf", "if_purc", "if_sc"), names(orig_data))
if (length(binary_flags) > 0) orig_data[, (binary_flags) := lapply(.SD, function(x) coalesce(x, 0L)), .SDcols = binary_flags]

# --- 8. LOAN AGE ENGINEERING -------------------------------------------------
# To create continues clean_loan_age (Thesis 3.3.2)
orig_date_lookup <- orig_data[, .(Loan_Sequence_Number, First_Pmt_Date)]
perf_data[orig_date_lookup, true_orig_date := i.First_Pmt_Date, on = "Loan_Sequence_Number"]
perf_data[, clean_loan_age :=
            (floor(Monthly_Reporting_Period / 100) - floor(true_orig_date / 100)) * 12 +
            (Monthly_Reporting_Period %% 100 - true_orig_date %% 100)]
perf_data[, true_orig_date := NULL]

# --- 9. FINAL MERGE & SAVE ---------------------------------------------------
final_cols <- intersect(c("Loan_Sequence_Number", ORIG_NUMERIC_COLS, binary_flags, "Geo_Key", "Lender_Key"), names(orig_data))
final_features <- merge(perf_data, orig_data[, .SD, .SDcols = final_cols], by = "Loan_Sequence_Number", all.x = TRUE)
final_features[, delq_numeric := NULL]

saveRDS(final_features, file.path(SAVE_DIR, "final_features_raw_clean.rds"))