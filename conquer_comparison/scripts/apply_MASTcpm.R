suppressPackageStartupMessages(library(MAST))
suppressPackageStartupMessages(library(edgeR))

run_MASTcpm <- function(L) {
  message("MAST, CPM")
  session_info <- sessionInfo()
  timing <- system.time({
    stopifnot(all(names(L$condt) == colnames(L$count)))
    grp <- L$condt
    dge <- DGEList(counts = L$count)
    dge <- edgeR::calcNormFactors(dge)
    cpms <- edgeR::cpm(dge)
    sca <- FromMatrix(exprsArray = log2(cpms + 1), 
                      cData = data.frame(grp = grp))
    zlmdata <- zlm(~grp, sca)

    summaryCond <- summary(zlmdata, doLRT=TRUE)
    summaryDt <- summaryCond$datatable
    
    contrasts_levels <- levels(summaryDt$contrast)
    str_filter <- function(str) startsWith(str, prefix="grp")
    admissible_keys <- Filter(str_filter, contrasts_levels)
    admissible_key <- admissible_keys[1]
    fcHurdle <- merge(
                    summaryDt[contrast==admissible_key & component=='H',.(primerid, `Pr(>Chisq)`)],
                        #hurdle P values
                    summaryDt[contrast==admissible_key & component=='logFC', .(primerid, coef, ci.hi, ci.lo)],
                    by='primerid' #logFC coefficients
    )
    df = data.frame(pval = fcHurdle$"Pr(>Chisq)",
                lfc = fcHurdle$coef,
                row.names = fcHurdle$primerid)
    df <- df[order(as.numeric(row.names(df))),]

  })
  
  list(session_info = session_info,
       timing = timing,
       df = df)
}

run_MASTcpm_multibatch <- function(L) {
  message("MAST, CPM MULTI BATCHES")
  session_info <- sessionInfo()
  timing <- system.time({
    stopifnot(all(names(L$condt) == colnames(L$count)))
    condition <- L$condt
    batch <- L$batch
    dge <- DGEList(counts = L$count)
    dge <- edgeR::calcNormFactors(dge)
    cpms <- edgeR::cpm(dge)
    sca <- FromMatrix(exprsArray = log2(cpms + 1), 
                      cData = data.frame(condition = condition))
    
    zlmdata <- zlm(~condition + batch, sca)
    summaryCond <- summary(zlmdata, doLRT="condition1")
    
    summaryDt <- summaryCond$datatable
    contrasts_levels <- levels(summaryDt$contrast)
    str_filter <- function(str) startsWith(str, prefix="L$condt")
    admissible_keys <- Filter(str_filter, contrasts_levels)
    admissible_key <- admissible_keys[1]

    fcHurdle <- merge(
                    # summaryDt[contrast==admissible_key & component=='H',.(primerid, `Pr(>Chisq)`)],
                    summaryDt[contrast=="condition1" & component=='H',.(primerid, `Pr(>Chisq)`)],
                        #hurdle P values
                    # summaryDt[contrast==admissible_key & component=='logFC', .(primerid, coef, ci.hi, ci.lo)],
                    summaryDt[contrast=="condition1" & component=='logFC', .(primerid, coef, ci.hi, ci.lo)],
                    by='primerid' #logFC coefficients
    )
    df = data.frame(pval = fcHurdle$"Pr(>Chisq)",
                lfc = fcHurdle$coef,
                row.names = fcHurdle$primerid)
    df <- df[order(as.numeric(row.names(df))),]
    # mast <- lrTest(zlmdata, "grp")
    # lfcs <- getLogFC(zlmdata)
  })
  
  list(session_info = session_info,
       timing = timing,
       df = df)
}
