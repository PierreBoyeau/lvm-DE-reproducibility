suppressPackageStartupMessages(library(limma))
suppressPackageStartupMessages(library(edgeR))

run_voomlimma <- function(L) {
  message("voomlimma")
  session_info <- sessionInfo()
  timing <- system.time({
    dge <- DGEList(L$count, group = L$condt)
    dge <- calcNormFactors(dge)
    condt <- factor(make.names(L$condt))
    design <- model.matrix(~condt)
    vm <- voom(dge, design = design, plot = TRUE)
    fit <- lmFit(vm, design = design)

    str_filter <- function(str) startsWith(str, prefix="condt")
    contrasts_levels <- colnames(coef(fit))
    admissible_keys <- Filter(str_filter, contrasts_levels)
    admissible_key <- c(admissible_keys[[1]])

    print(admissible_key)
    print(colnames(coef(fit)))
    # do.call("<-",list(admissible_key, admissible_key))


    # contr <- makeContrasts("condtX1 - (Intercept)", levels = colnames(coef(fit)))    
    contr <- makeContrasts("condtX1", levels = colnames(coef(fit)))    
    fit <- contrasts.fit(fit, contr)
    fit <- eBayes(fit)
    tt <- topTable(fit, n = Inf, adjust.method = "BH", sort.by = "none")
  })
  
  tt <- tt[order(as.numeric(row.names(tt))),]
  # hist(tt$P.Value, 50)
  # hist(tt$adj.P.Val, 50)
  # limma::plotMDS(dge, col = as.numeric(as.factor(L$condt)), pch = 19)
  # plotMD(fit)
  
  list(session_info = session_info,
       timing = timing,
       tt = tt,
       df = data.frame(pval = tt$P.Value,
                       padj = tt$adj.P.Val,
                       lfc = tt$logFC,
                       row.names = rownames(tt)))
}


run_voomlimma_multibatch <- function(L) {
  message("voomlimma multibatch")
  session_info <- sessionInfo()
  timing <- system.time({
    dge <- DGEList(L$count, group = L$condt)
    dge <- calcNormFactors(dge)
    condt <- factor(make.names(L$condt))
    batch <- factor(make.names((L$batch)))
    design <- model.matrix(~condt + batch)
    vm <- voom(dge, design = design, plot = TRUE)
    fit <- lmFit(vm, design = design)

    str_filter <- function(str) startsWith(str, prefix="condt")
    contrasts_levels <- colnames(coef(fit))
    admissible_keys <- Filter(str_filter, contrasts_levels)
    admissible_key <- c(admissible_keys[[1]])

    print(admissible_key)
    print(colnames(coef(fit)))
    # do.call("<-",list(admissible_key, admissible_key))

    # contr <- makeContrasts("condtX1 - (Intercept)", levels = colnames(coef(fit)))
    contr <- makeContrasts("condtX1", levels = colnames(coef(fit)))
    fit <- contrasts.fit(fit, contr)
    fit <- eBayes(fit)
    tt <- topTable(fit, n = Inf, adjust.method = "BH")
  })

  tt <- tt[order(as.numeric(row.names(tt))),]
  hist(tt$P.Value, 50)
  hist(tt$adj.P.Val, 50)
  limma::plotMDS(dge, col = as.numeric(as.factor(L$condt)), pch = 19)
  plotMD(fit)
  
  list(session_info = session_info,
       timing = timing,
       tt = tt,
       df = data.frame(pval = tt$P.Value,
                       padj = tt$adj.P.Val,
                       lfc = tt$logFC,
                       row.names = rownames(tt)))
}
