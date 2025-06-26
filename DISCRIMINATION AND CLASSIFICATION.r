library(dplyr)
library(MASS)      
#install.packages("caret")
library(caret)     
library(ggplot2)   
library(e1071)
library(pROC)

data <- read.csv("en_en_en_son_data.csv")
head(data)
data <- data %>% select(-NumberOfOpenCreditLines, -NumberOfCreditInquiries)
print(colnames(data))

# Selecting important variables
selected_columns <- c("TotalAssets", "CreditScore", "RiskScore", "TotalLiabilities",
                      "SavingsAccountBalance", "CheckingAccountBalance", 
                      "LoanAmount", "LoanApproved")
selected_columns2 <- c("TotalAssets", "CreditScore", "RiskScore", "TotalLiabilities","SavingsAccountBalance", "CheckingAccountBalance","LoanAmount")
data_selected <- data[selected_columns]

#Quantifying categorical variables(label encoding)
categorical_columns <- sapply(data, is.factor) | sapply(data, is.character)

for (col in names(data)[categorical_columns]) {
  data[[col]] <- as.numeric(as.factor(data[[col]]))
}


# Splitting the data into 80% training and 20% testing
set.seed(42)
train_index <- createDataPartition(data_selected$LoanApproved, p = 0.8, list = FALSE)
train_data <- data_selected[train_index, ]
test_data <- data_selected[-train_index, ]

# LDA Model Training

lda_model <- lda(LoanApproved ~ ., data = train_data)
print(lda_model)
plot(lda_model)

lda_predictions <- predict(lda_model, test_data)
names(lda_predictions)


#Confusion Matrix)
confusion_matrix <- table(Predicted = lda_predictions$class, Actual = test_data$LoanApproved)
print(confusion_matrix)



confusion <- confusionMatrix(lda_predictions$class, as.factor(test_data$LoanApproved))
print(confusion)

# Accuracy, Precision, Recall, F1 Score
accuracy <- confusion$overall['Accuracy']
precision <- confusion$byClass['Precision']
recall <- confusion$byClass['Recall']
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")


# Visualization for component LD1
lda_data <- data.frame(LD1 = lda_predictions$x[, 1], Actual = test_data$LoanApproved)

ggplot(lda_data, aes(x = LD1, fill = as.factor(Actual))) +
  geom_histogram(alpha = 0.6, bins = 30, position = "identity") +
  labs(title = "LDA - Separation of Groups", x = "First Linear Separator (LD1)", y = "Frequency") +
  theme_minimal() +
  scale_fill_discrete(name = "Real Class")


#install.packages("klaR")
library(klaR)

#Set the target variable as the factor
train_data$LoanApproved <- as.factor(train_data$LoanApproved)


# Partition Plot using
png("partition_plot.png", width = 2000, height = 1500)  # Görselleştirme çıktısını kaydetmek için
partimat(LoanApproved ~ TotalAssets + CreditScore , , data = train_data, method = "lda")


partimat(LoanApproved ~ TotalAssets  + RiskScore , data = train_data, method = "lda")

partimat(LoanApproved ~ TotalAssets  +TotalLiabilities , data = train_data, method = "lda")

partimat(LoanApproved ~ TotalAssets + 
         SavingsAccountBalance   , data = train_data, method = "lda")

partimat(LoanApproved ~ TotalAssets + CreditScore + RiskScore + TotalLiabilities + 
         SavingsAccountBalance + CheckingAccountBalance + LoanAmount, , data = train_data, method = "lda")


partimat(LoanApproved ~ TotalAssets + CreditScore + RiskScore + TotalLiabilities + 
         SavingsAccountBalance + CheckingAccountBalance + LoanAmount, , data = train_data, method = "lda")

train_data$LoanApproved <- as.factor(train_data$LoanApproved)

# Modeli Eğitme
lda_model <- lda(LoanApproved ~ TotalAssets + CreditScore, data = train_data)

# Tahmin yapmak için bir grid oluşturma
x_seq <- seq(min(train_data$TotalAssets), max(train_data$TotalAssets), length.out = 100)
y_seq <- seq(min(train_data$CreditScore), max(train_data$CreditScore), length.out = 100)
grid <- expand.grid(TotalAssets = x_seq, CreditScore = y_seq)

# Grid üzerinde tahmin yapma
grid$Prediction <- predict(lda_model, newdata = grid)$class

# Sınıflandırma sınırlarının görselleştirilmesi
ggplot() +
  geom_tile(data = grid, aes(x = TotalAssets, y = CreditScore, fill = Prediction), alpha = 0.5) +
  geom_point(data = train_data, aes(x = TotalAssets, y = CreditScore, color = LoanApproved)) +
  labs(title = "LDA Classification Boundaries", x = "Total Assets", y = "Credit Score") +
  theme_minimal()

library(ggplot2)
library(gridExtra)
library(MASS)

# LDA Modeli için seçilen veriyi hazırlayın
selected_columns <- c("TotalAssets", "CreditScore", "RiskScore", "TotalLiabilities",
                      "SavingsAccountBalance", "CheckingAccountBalance", 
                      "LoanAmount", "LoanApproved")
data_selected <- data[selected_columns]

# Kategorik değişkenleri sayısal hale getirme (Gerekliyse)
categorical_columns <- sapply(data_selected, is.factor) | sapply(data_selected, is.character)
for (col in names(data_selected)[categorical_columns]) {
  data_selected[[col]] <- as.numeric(as.factor(data_selected[[col]]))
}

# Eğitim ve test verisine ayırma
set.seed(42)
train_index <- createDataPartition(data_selected$LoanApproved, p = 0.8, list = FALSE)
train_data <- data_selected[train_index, ]
train_data$LoanApproved <- as.factor(train_data$LoanApproved)

# Değişken çiftleri için kombinasyonlar oluşturma
feature_names <- names(data_selected)[names(data_selected) != "LoanApproved"]
combinations <- combn(feature_names, 2, simplify = FALSE)

# Her kombinasyon için grafik oluşturma
plots <- lapply(combinations, function(vars) {
  # Vars: Şu anda işlenen iki değişken
  var_x <- vars[1]
  var_y <- vars[2]
  
  # LDA Modeli
  lda_model <- lda(LoanApproved ~ ., data = train_data[, c(var_x, var_y, "LoanApproved")])
  
  # Grid için tahminler
  x_seq <- seq(min(train_data[[var_x]]), max(train_data[[var_x]]), length.out = 100)
  y_seq <- seq(min(train_data[[var_y]]), max(train_data[[var_y]]), length.out = 100)
  grid <- expand.grid(X = x_seq, Y = y_seq)
  colnames(grid) <- c(var_x, var_y)
  grid$Prediction <- predict(lda_model, newdata = grid)$class
  
  # Grafik
  p <- ggplot() +
    geom_tile(data = grid, aes_string(x = var_x, y = var_y, fill = "Prediction"), alpha = 0.5) +
    geom_point(data = train_data, aes_string(x = var_x, y = var_y, color = "LoanApproved")) +
    labs(title = paste("LDA:", var_x, "vs", var_y),
         x = var_x, y = var_y) +
    theme_minimal()
  return(p)
})

# Tüm grafikleri birleştirme
grid.arrange(grobs = plots, ncol = 3)
 

















library(ggplot2)
library(gridExtra)
library(MASS)

# Prepare the selected data for the LDA model.
selected_columns <- c("TotalAssets", "CreditScore", "RiskScore", "TotalLiabilities",
                      "SavingsAccountBalance", "CheckingAccountBalance", 
                      "LoanAmount", "LoanApproved")
data_selected <- data[selected_columns]

#Convert categorical variables to numerical format (if necessary).
categorical_columns <- sapply(data_selected, is.factor) | sapply(data_selected, is.character)
for (col in names(data_selected)[categorical_columns]) {
  data_selected[[col]] <- as.numeric(as.factor(data_selected[[col]]))
}

#Split into training and testing data.
set.seed(42)
train_index <- createDataPartition(data_selected$LoanApproved, p = 0.8, list = FALSE)
train_data <- data_selected[train_index, ]
train_data$LoanApproved <- as.factor(train_data$LoanApproved)

# Creating combinations for variable pairs.
feature_names <- names(data_selected)[names(data_selected) != "LoanApproved"]
combinations <- combn(feature_names, 2, simplify = FALSE)

# Creating a plot for each combination.
plots <- lapply(combinations, function(vars) {
  # Vars: The two variables currently being processed.
  var_x <- vars[1]
  var_y <- vars[2]
  
  # LDA Model
  lda_model <- lda(LoanApproved ~ ., data = train_data[, c(var_x, var_y, "LoanApproved")])
  
  # Predictions for the grid.
  x_seq <- seq(min(train_data[[var_x]]), max(train_data[[var_x]]), length.out = 100)
  y_seq <- seq(min(train_data[[var_y]]), max(train_data[[var_y]]), length.out = 100)
  grid <- expand.grid(X = x_seq, Y = y_seq)
  colnames(grid) <- c(var_x, var_y)
  grid$Prediction <- predict(lda_model, newdata = grid)$class
  
  # Plot
  p <- ggplot() +
    geom_tile(data = grid, aes_string(x = var_x, y = var_y, fill = "Prediction"), alpha = 0.5) +
    geom_point(data = train_data, aes_string(x = var_x, y = var_y, color = "LoanApproved")) +
    labs(x = var_x, y = var_y) +  
    theme_minimal() +
    theme(plot.title = element_blank())  
  return(p)
})

# Merge all plots
grid.arrange(grobs = plots, ncol = 3)































#Let’s look at the plots individually.
#Multiple chart layout (4 rows, 5 columns) and margins
par(mfrow = c(4, 5), mar = c(2, 2, 2, 1))

#Get the names of all arguments (except the target variable)
variables <- names(train)[-which(names(train) == "LoanApproved")]

# Implementing partimat for variable pairs with loop
for (i in 1:(length(variables) - 1)) {
  for (j in (i + 1):length(variables)) {
    # Create formua for two variables
    formula <- as.formula(paste("LoanApproved ~", variables[i], "+", variables[j]))
    # partimat application
    partimat(formula, data = train, method = "lda")
  }
}


#Train performance:

# Predicting the training data using the model
train_predict <- predict(lda_model, train_data)$class

#Creating Confusion Matrix for training data
table_train <- table(Predicted = train_predict, Actual = train_data$LoanApproved)
print(table_train)

#Calculate accuracy
sum(diag(table_train))/sum(table_train)

#Test Performance:

# Set the target variable as the factor
test_data$LoanApproved <- as.factor(test_data$LoanApproved)

# Predicting test data using the model
test_predict <- predict(lda_model, test_data)$class

# Creating Confusion Matrix for test data
table_test <- table(Predicted = test_predict, Actual = test_data$LoanApproved)
print(table_test)
#calculate accuracy
sum(diag(table_test))/sum(table_test)

#The model correctly classifies the patients with 0.99025 probability for the test data.
#The classification error rate (misclassification rate) for test data is 1-0.99025=0,0075

#10-fold cross-validation for check over-fitting
train_data$LoanApproved <- as.factor(train_data$LoanApproved)
cv_control <- trainControl(method = "cv", number = 10) # 10-fold Cross Validation
cv_model <- train(LoanApproved ~ ., data = train_data, method = "lda", trControl = cv_control)
print(cv_model)


#Check roc and auc score after cross-validation
# ROC Curve ve AUC Score
roc_curve <- roc(as.numeric(test_data$LoanApproved), as.numeric(test_predict))
plot(roc_curve, col = "blue", main = "ROC Curve")
auc_score <- auc(roc_curve)
cat("AUC Score: ", auc_score, "\n")

# Precision, Recall ve F1 Score Hesaplama
precision <- posPredValue(test_predict, test_data$LoanApproved, positive = "1")
recall <- sensitivity(test_predict, test_data$LoanApproved, positive = "1")
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")

# UC (Area Under the Curve) ~ 0.986
# Indicates the model's ability to distinguish between positive and negative classes. Closer to 1 means better. 0.986 is very high.

# Precision ~ 0.981
# Represents how many predicted positives are true positives. A high value (0.981) means a very low false positive rate.

# Recall (Sensitivity) ~ 0.978
# Shows the proportion of actual positives correctly identified. 0.978 indicates the model captures nearly all true positives.

# F1 Score ~ 0.980
# Harmonic mean of Precision and Recall. A high value (0.980) confirms the model balances both well.
