"""Build a dataset of TVA data.

This roughly follows MRV's process, originally written in SAS
`21_indicateurs.sas`. Source data should be stored beforehand inside an input directory
which, in turn, contains the following directories containing the data as (possibly
multiple) orc file(s):
- liasse_tva_ca3_view (etl_tva)
- liasse_tva_ca12_view (etl_tva)

USAGE
    python tva_dataset.py <tva_tables_directory> <output_directory>

"""

import os
import re
import sys
from os import path

import pyspark.sql.functions as F

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable = C0413
import sf_datalake.io
import sf_datalake.transform
import sf_datalake.utils

####################
# Loading datasets #
####################

parser = sf_datalake.io.data_path_parser()
parser.description = "Build a dataset of monthly/quarterly TVA data."
args = parser.parse_args()

data_paths = {
    "liasse_tva_ca3": path.join(args.input, "liasse_tva_ca3_view"),
    "liasse_tva_ca12": path.join(args.input, "liasse_tva_ca12_view"),
}
datasets = sf_datalake.io.load_data(data_paths, file_format="orc")

# Set every column name to lower case (if not already).
for name, ds in datasets.items():
    datasets[name] = ds.toDF(*(col.lower() for col in ds.columns))

# TVA can be declared either on a:
# - monthly,
# - quarterly,
# - yearly,
# basis
#
# We join data from different types of declarations and add the "duree_periode_tva"
# variable to describe the period duration parameter.
#

tva_join_columns = list(
    set(datasets["liasse_tva_ca3"].columns) & set(datasets["liasse_tva_ca12"].columns)
)
all_tva = (
    datasets["liasse_tva_ca3"]
    .join(datasets["liasse_tva_ca12"], on=tva_join_columns, how="outer")
    .withColumn(
        "duree_periode_tva",
        F.round(
            F.months_between(
                F.col("dte_fin_periode"),
                F.col("dte_debut_periode"),
            )
        ).cast("integer"),
    )
)
x_tva = all_tva.na.fill(value=0, subset=sf_datalake.utils.numerical_columns(all_tva))
x_tva = x_tva.withColumn("d_tca_total", F.col("d3310_29") + F.col("d3517s_55_i"))
x_tva = x_tva.withColumn(
    "d_tva_ni_b0032_export", F.col("d3517s_02_b") + F.col("d3310_04")
)
x_tva = x_tva.withColumn("d_tva_ni_b0034_lic", F.col("d3517s_04_b") + F.col("d3310_06"))
x_tva = x_tva.withColumn(
    "d_tva_ni_b0037_ach_frch", F.col("d3517s_01_b") + F.col("d3310_07")
)
x_tva = x_tva.withColumn(
    "d_tva_ni_b0029_liv_el_gaz", F.col("d3517s_4d_b") + F.col("d3310_6a")
)
x_tva = x_tva.withColumn(
    "d_tva_ni_b0043_assjt_hs_fr", F.col("d3517s_4b_b") + F.col("d3310_7a")
)
x_tva = x_tva.withColumn(
    "m_tva_ni_b0033_autr_op_ni",
    F.col("d3310_7b") + F.col("d3517s_03_b") + F.col("d3310_05"),
)
cols_SUM_TVA_NI_bTOTAL = [
    "d_tva_ni_b0032_export",
    "d_tva_ni_b0034_lic",
    "d_tva_ni_b0037_ach_frch",
    "d_tva_ni_b0029_liv_el_gaz",
    "d_tva_ni_b0043_assjt_hs_fr",
    "m_tva_ni_b0033_autr_op_ni",
]
x_tva = x_tva.withColumn(
    "sum_tva_ni_btotal", sum(x_tva[col] for col in cols_SUM_TVA_NI_bTOTAL)
)
cols_M_TVA_BI_b0979_CA = [
    "d3310_01",  # "d3517s_05_b0206",
    "d3517s_5a_b",
    "d3517s_06_b",  # "d3517s_6b_b0150",
    "d3517s_6c_b",
    "d3517s_07_b",
    "d3517s_08_b",
    "d3517s_09_b",
    "d3517s_10_b",
]
x_tva = x_tva.withColumn(
    "m_tva_bi_b0979_ca", sum(x_tva[col] for col in cols_M_TVA_BI_b0979_CA)
)
cols_M_TVA_BI_b0981_AUTR_OP_IMP = [
    "d3310_02",  # "d3310_2b",
    "d3310_3c",
    "d3517s_13_b",
    "d3517s_11_b",
    "d3517s_12_b",
]
x_tva = x_tva.withColumn(
    "m_tva_bi_b0981_autr_op_imp",
    sum(x_tva[col] for col in cols_M_TVA_BI_b0981_AUTR_OP_IMP),
)
x_tva = x_tva.withColumn(
    "d_tva_bi_b0044_ach_ps_ic", F.col("d3517s_ac_b") + F.col("d3310_2a")
)
x_tva = x_tva.withColumn("d_tva_bi_b0031_aic", F.col("d3517s_14_b") + F.col("d3310_03"))
x_tva = x_tva.withColumn(
    "d_tva_bi_b0030_liv_el_gaz", F.col("d3517s_aa_b") + F.col("d3310_3a")
)
x_tva = x_tva.withColumn(
    "d_tva_bi_b0040_assjt_hs_fr", F.col("d3517s_ab_b") + F.col("d3310_3b")
)
cols_SUM_TVA_BI_bTOTAL = [
    "m_tva_bi_b0979_ca",
    "m_tva_bi_b0981_autr_op_imp",
    "d_tva_bi_b0044_ach_ps_ic",
    "d_tva_bi_b0031_aic",
    "d_tva_bi_b0030_liv_el_gaz",
    "d_tva_bi_b0040_assjt_hs_fr",
]
x_tva = x_tva.withColumn(
    "sum_tva_bi_btotal", sum(x_tva[col] for col in cols_SUM_TVA_BI_bTOTAL)
)
x_tva = x_tva.withColumn(
    "sum_tva_ni_bi_btotal", F.col("sum_tva_bi_btotal") + F.col("sum_tva_ni_btotal")
)
cols_M_TVA_BI_b0207_NORMAL = [  # "d3517s_05_b0206",
    "d3310_08_btx196",
    "d3517s_5a_b",
    "d3310_08_b",
    "d3517s_11_b",
    "d3517s_12_b",
    "d3517s_13_b",
    "d3517s_14_b",
    "d3517s_ab_b",
    "d3517s_ac_b",
    "d3517s_aa_b",
]
x_tva = x_tva.withColumn(
    "m_tva_bi_b0207_normal", sum(x_tva[col] for col in cols_M_TVA_BI_b0207_NORMAL)
)
x_tva = x_tva.withColumn(
    "m_tva_bi_b0105_reduit_5_5", F.col("d3517s_06_b") + F.col("d3310_09_b")
)
cols_M_TVA_BI_b0151_REDUIT_10 = [  # D3517S_6B_b0150,
    "d3310_9b_btx7",
    "d3517s_6c_b",
    "d3310_9b_b",
]
x_tva = x_tva.withColumn(
    "m_tva_bi_b0151_reduit_10", sum(x_tva[col] for col in cols_M_TVA_BI_b0151_REDUIT_10)
)
x_tva = x_tva.withColumn(
    "m_tva_bi_b0100_dom_2_1", F.col("d3517s_08_b") + F.col("d3310_11_b")
)
x_tva = x_tva.withColumn(
    "m_tva_bi_b0201_dom_8_5", F.col("d3517s_07_b") + F.col("d3310_10_b")
)
x_tva = x_tva.withColumn(
    "d_tva_bi_b0950_tx_part", F.col("d3517s_09_b") + F.col("d3310_14_b")
)
x_tva = x_tva.withColumn(
    "m_tva_bi_b0900_anc_tx", F.col("d3517s_10_b") + F.col("d3310_13_b")
)
x_tva = x_tva.withColumn(
    "d_tva_col_i0600_ant_ded", F.col("d3310_15") + F.col("d3517s_18_i")
)
x_tva = x_tva.withColumn(
    "sum_tva_col_total",
    F.col("d3310_16")
    - F.col("d3310_15")
    + F.col("d3517s_16_i")
    - F.col("d3310_7c")
    - F.col("d3310_17")
    - F.col("d3310_5b")
    - F.col("d3517s_aa_i")
    - F.col("d3517s_ab_i")
    - F.col("d3517s_ac_i")
    - F.col("d3517s_13_i")
    - F.col("d3517s_14_i"),
)
x_tva = x_tva.withColumn(
    "d_tva_col_i0031_aic", F.col("d3517s_14_i") + F.col("d3310_17")
)
x_tva = x_tva.withColumn(
    "d_tva_ded_i0703_imm", F.col("d3310_19") + F.col("d3517s_23_i")
)
x_tva = x_tva.withColumn(
    "m_tva_ded_i0702_abs",
    F.col("d3310_20") + F.col("d3517s_20_i") + F.col("d3517s_21_i"),
)
x_tva = x_tva.withColumn(
    "d_tva_ded_i0059_autr", F.col("d3310_21") + F.col("d3517s_25_i")
)  # D3310_2C

x_tva = x_tva.withColumn(
    "d_tva_ded_tx_coef_ded",
    F.when(
        (F.col("d3310_22a") == 0.0) & (F.col("D3517S_25a_tx_ded") == 0.0), 100
    ).otherwise(F.col("d3310_22a") + F.col("D3517S_25a_tx_ded")),
)
x_tva = x_tva.withColumn(
    "d_tva_ded_i0705_total", F.col("d3310_23") + F.col("d3517s_26_i")
)
x_tva = x_tva.withColumn(
    "d_tva_ded_total_hs_report",
    F.col("d_tva_ded_i0703_imm")
    + F.col("m_tva_ded_i0702_abs")
    + F.col("d_tva_ded_i0059_autr"),
)
x_tva = x_tva.withColumn(
    "d_tva_ded_i0709_dt_es_dom", F.col("d3310_24") + F.col("d3517s_27_i")
)
x_tva = x_tva.withColumn(
    "m_tva_net_i8002_remb_dem", F.col("d3310_26") + F.col("d3517s_50_i")
)
x_tva = x_tva.withColumn("m_tva_net_due", F.col("d3310_28") + F.col("d3517s_28_i"))

# Write file
raw_cols_re = re.compile("d3310*|d3517*")
output_tva = x_tva.drop(
    *(
        [col for col in x_tva.columns if raw_cols_re.match(col)]
        + ["no_ocfi", "mode_depot", "version_form", "dte_depot"]
    )
)
output_tva.withColumnRenamed("dte_debut_periode", "date_deb_tva").withColumnRenamed(
    "dte_fin_periode", "date_fin_tva"
).write.format("orc").save(path.join(args.output, "tva"))
