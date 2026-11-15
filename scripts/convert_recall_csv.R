library(utils)

# ler o CSV do recordatorio (ja extraido)
csv_path <- "C:/Users/I555769/OneDrive - SAP SE/Desktop/TCC/GAIA/data/raw/enani_recall_24h.csv"
out_path <- "C:/Users/I555769/OneDrive - SAP SE/Desktop/TCC/GAIA/data/raw/enani_recall_24h_clean.csv"

df <- read.csv(csv_path, stringsAsFactors = FALSE)

cat("Shape:", nrow(df), "x", ncol(df), "\n")
cat("Colunas:", paste(names(df), collapse=", "), "\n\n")

write.csv(df, out_path, row.names = FALSE, fileEncoding = "UTF-8")
cat("Salvo:", out_path, "\n")
