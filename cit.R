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
library(ggplot2)
library(caret)
library(tibble)
library(cvms)

# Get credit data from openml
odata <- odt(id = 31)
df <- odata$data

##########################
# Exploratory Data Analysis
##########################

# Class Distribution
ggplot(df, aes(x = class, fill = class)) + 
  geom_bar() +
  labs(title = "Class Distribution", x = "Class", y = "Count") +
  theme_minimal()

# Credit History Distribution
ggplot(df, aes(x = credit_history, fill = credit_history)) +
  geom_bar() +
  labs(title = "Credit History ",x = "Type",y = "Count") +
  theme_minimal()

# Histogram of Age
hist(df$age, col=rgb(0.2,0.8,0.5,0.5), main="" , xlab="Age")
abline(v = mean(df$age), col="red", lwd=3, lty=2)


##########################
# Visualize CIT
##########################

cit <- party::ctree(class ~ ., data = df)

#plot(cit)
#party::nodes(cit, 1)[[1]]$criterion$criterion
#party::nodes(cit, 2)[[1]]$criterion$criterion

##########################
# Calculate Evaluation Metrics
##########################

calculate_metrics <- function(true_labels, predicted_labels) {
  # Create confusion matrix
  conf_matrix <- table(predicted_labels, true_labels)
  print(conf_matrix)
  
  # Calculate metrics
  TP <- conf_matrix[2, 2]  # True positives
  FP <- conf_matrix[2, 1]  # False positives
  FN <- conf_matrix[1, 2]  # False negatives
  TN <- conf_matrix[1, 1]  # True negatives
  
  # Calculate metrics
  accuracy <- (TP + TN) / sum(conf_matrix)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # Store metrics in a dataframe
  metrics_df <- data.frame(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  )
  
  return(metrics_df)
}

##########################
# Build model with the original data set
##########################

# Data splitting
set.seed(1)
train <- caret::createDataPartition(df$class, p = 0.70, list = FALSE)
train.data <- df[train,]
test.data <- df[-train,]


# Baseline Results
baseline.cit <- party::ctree(class ~ ., data = train.data, controls = ctree_control(testtype = "Bonferroni"))
plot(baseline.cit)
pred <- predict(baseline.cit, test.data)
confusionMatrix(test.data$class, pred)
baseline.results <- calculate_metrics(test.data$class, pred)

##########################
# Tuned Model
##########################
# 10-fold cross validation
ctrl <- caret::trainControl(method = "cv",number = 10, summaryFunction = multiClassSummary)
cit.kf <- caret::train(class ~ ., data = train.data, 
                       method = "ctree", 
                       trControl = ctrl,
                       controls = ctree_control(testtype = "Bonferroni"),
                       tuneLength = 50)
cit.kf # results
plot(cit.kf,xlab="P-Value Threshold") # plots cv graph 
pred <- predict(cit.kf, test.data) # Cross-validated model
confusionMatrix(test.data$class, pred) # Confusion Matrix
tuned.model.results <- calculate_metrics(test.data$class, pred)

# Plot Confusion Matrix
cfm <- as_tibble(table(tibble("target" = test.data$class,
                           "prediction" = pred)))
plot_confusion_matrix(cfm, 
                      target_col = "target", 
                      prediction_col = "prediction",
                      counts_col = "n")

# Plot the Conditional Inference Tree
tree <- party::ctree(class ~ ., data = train.data, 
                     controls = ctree_control(mincriterion = 0.83, testtype = "Bonferroni"))
plot(tree) 
tree

##########################
# Build Model with downsampled data set
##########################
df_down <- caret::downSample(x = df[, -c("class")], 
                             y = df$class)
colnames(df_down)[length(df_down)] <- "class"

cit.down <- party::ctree(class ~ ., data = df_down)

plot(cit.down)

# Data splitting
set.seed(1)
train <- caret::createDataPartition(df_down$class, p = 0.70, list = FALSE)
train.data <- df_down[train,]
test.data <- df_down[-train,]

# 10-fold cross validation with downsampled data
cit.dkf <- caret::train(class ~ ., data = train.data, 
                        method = "ctree", 
                        trControl = ctrl,
                        tuneLength = 50)
cit.dkf # results
plot(cit.dkf, xlab="P-value Threshold") # Plots cv graph

pred <- predict(cit.dkf, test.data) # Predictions
confusionMatrix(test.data$class, pred)
cit.dkf.results <- calculate_metrics(test.data$class, pred)

# Plot Confusion Matrix
cfm <- as_tibble(table(tibble("target" = test.data$class,
                              "prediction" = pred)))
plot_confusion_matrix(cfm, 
                      target_col = "target", 
                      prediction_col = "prediction",
                      counts_col = "n")




#res <- as.data.frame(nodes(cit, 2)[[1]]$criterion$criterion)
#col_names <- row.names(res)
#res$var <- col_names
#res <- as.data.frame(res[order(res$`nodes(cit, 2)[[1]]$criterion$criterion`, decreasing=TRUE),])



