library(utils)

zip_path <- "C:/Users/I555769/OneDrive - SAP SE/Desktop/TCC/enani-banco-de-dados-2019-v2.zip"
out_dir  <- "C:/Users/I555769/OneDrive - SAP SE/Desktop/TCC/GAIA/data/raw"
tmp_dir  <- tempdir()

unzip(zip_path, files = "Banco de dados/data_crianca_calib_anon.rds", exdir = tmp_dir)
full_path <- file.path(tmp_dir, "Banco de dados/data_crianca_calib_anon.rds")

obj <- readRDS(full_path)
df  <- as.data.frame(obj$variables)

cat("Shape:", nrow(df), "x", ncol(df), "\n")
cat("Colunas:", paste(names(df), collapse=", "), "\n\n")

write.csv(df, file.path(out_dir, "enani_profiles.csv"), row.names = FALSE)
cat("Salvo: enani_profiles.csv\n")
