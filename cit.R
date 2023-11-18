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
library(party)
library(sandwich)

# Get credit data from openml
odata <- odt(id = 31)
df <- odata$data

cit <- party::ctree(class ~ ., data = df)

plot(cit)

party::nodes(cit, 1)[[1]]$criterion$criterion
party::nodes(cit, 2)[[1]]$criterion$criterion


##########################
set.seed(1)
train <- caret::createDataPartition(df$class, p = 0.70, list = FALSE)
train.data <- df[train,]
test.data <- df[-train,]
# 10-fold cross validation
ctrl <- caret::trainControl(method = "cv",number = 10)
cit.kf <- caret::train(class ~ ., data = train.data, 
                       method = "ctree", 
                       trControl = ctrl)
cit.kf # results
plot(cit.kf) # plots cv graph 
pred <- predict(cit.kf, test.data) # Cross-validated model
confusionMatrix(test.data$class, pred)

tree <- ctree(class ~ ., data = train.data, 
                     controls = ctree_control(mincriterion = 0.95))
plot(tree) # plot the tree
tree
###########################

# Downsampling
df_down <- caret::downSample(x = df[, -c("class")], 
                             y = df$class)
colnames(df_down)[length(df_down)] <- "class"

cit.down <- party::ctree(class ~ ., data = df_down)

plot(cit.down)

# 10-fold cross validation with downsampled data

set.seed(1)
train <- caret::createDataPartition(df_down$class, p = 0.70, list = FALSE)
train.data <- df_down[train,]
test.data <- df_down[-train,]

ctrl <- caret::trainControl(method = "cv",number = 10)
cit.dkf <- caret::train(class ~ ., data = train.data, method = "ctree", trControl = ctrl)
cit.dkf # results
plot(cit.dkf) # plots cv graph

pred <- predict(cit.dkf, test.data)
confusionMatrix(test.data$class, pred)





#res <- as.data.frame(nodes(cit, 2)[[1]]$criterion$criterion)
#col_names <- row.names(res)
#res$var <- col_names
#res <- as.data.frame(res[order(res$`nodes(cit, 2)[[1]]$criterion$criterion`, decreasing=TRUE),])



