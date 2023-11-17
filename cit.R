if (!requireNamespace("mlr3oml", quietly = TRUE)) {
  install.packages("mlr3oml")
}
if (!requireNamespace("mlr3", quietly = TRUE)) {
  install.packages("mlr3")
}
if (!requireNamespace("DMwR", quietly = TRUE)) {
  install.packages("DMwR")
}
if (!requireNamespace("h2o", quietly = TRUE)) {
  install.packages("h2o")
}

library(mlr3oml)
library(mlr3)

# Get credit data from openml
odata = odt(id = 31)
df = odata$data
#write.csv(df, "credit.csv")
cit <- party::ctree(class ~ ., data = df)

plot(cit)

party::nodes(cit, 1)[[1]]$criterion$criterion
party::nodes(cit, 2)[[1]]$criterion$criterion

# 10-fold cross validation
ctrl <- caret::trainControl(method = "cv",number = 10)
cit.kf <- caret::train(class ~ ., data = df, method = "ctree", trControl = ctrl)
cit.kf # results

###########################

# Downsampling
df_down <- caret::downSample(x = df[, -c("class")], 
                             y = df$class)
colnames(df_down)[length(df_down)] <- "class"

cit.down <- party::ctree(class ~ ., data = df_down)

plot(cit.down)

# 10-fold cross validation with downsampled data
ctrl <- caret::trainControl(method = "cv",number = 10)
cit.dkf <- caret::train(class ~ ., data = df_down, method = "ctree", trControl = ctrl)
cit.dkf






#res <- as.data.frame(nodes(cit, 2)[[1]]$criterion$criterion)
#col_names <- row.names(res)
#res$var <- col_names
#res <- as.data.frame(res[order(res$`nodes(cit, 2)[[1]]$criterion$criterion`, decreasing=TRUE),])



# Automated Machine Learning
library(h2o)
h2o.init()

df <- h2o.importFile("credit.csv")
y <- "class"
x <- setdiff(names(df), y)

df[, y] <- as.factor(df[, y])

# Run AutoML for 20 base models
aml <- h2o.automl(x = x, y = y,
                  training_frame = df,
                  max_models = 20,
                  seed = 1)

lb <- aml@leaderboard
print(lb)
best_model <- h2o.get_best_model(aml)


m <- h2o.get_best_model(aml, criterion = "accuracy")
