# set work directory
getwd()
setwd("E:\\Study\\Chenkai GUO\\硕士\\概率图模型2024Summer")

# install and load required packages
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install(c("graph", "RBGL", "Rgraphviz"))
install.packages("pcalg")
install.packages("bnlearn")
install.packages("ggplot2")
install.packages("cowplot")

library(pcalg)
library(bnlearn)
library(ggplot2)
library(ggsci)
library(cowplot)
library(igraph)
library(Rgraphviz)


# generate simulated data
set.seed(123)
n <- 3000  # samples
p <- 15    # variables

randomDAG <- randomDAG(p, prob = 0.3)
adj_matrix <- as(randomDAG, "matrix")
data <- rmvDAG(n, randomDAG, errDist = "normal")

# scatter plot
pairs(data, main = "Scatterplot Matrix of Simulated Data")


# the following algorithms are all implemented by bnlearn package
# PC Algorithm
pc.time <- system.time({
  suffStat <- list(C = cor(data), n = n)
  pc.fit <- pc(suffStat, indepTest = gaussCItest, p = p, alpha = 0.01)
  pc_adj <- as(pc.fit@graph, "matrix")
})
pc_metrics <- evaluate_model(adj_matrix, pc_adj)

# transform to dataframe
data2 = as.data.frame(data)

# Hill-Climbing Algorithm
hc.time <- system.time({
  hc.fit <- hc(data2)
  hc_adj <- amat(hc.fit)
})
hc_metrics <- evaluate_model(adj_matrix, hc_adj)

# Greedy Algorithm
gs.time <- system.time({
  gs.fit <- gs(data2)
  gs_adj <- amat(gs.fit)
})
gs_metrics <- evaluate_model(adj_matrix, gs_adj)


# computation cost visualization
cat("PC Algorithm Time: ", pc.time[3], " seconds\n")
cat("Hill-Climbing Algorithm Time: ", hc.time[3], " seconds\n")
cat("Greedy Search Algorithm Time: ", gs.time[3], " seconds\n")

time_data <- data.frame(
  Algorithm = c("PC Algorithm", "Hill-Climbing", "Greedy Search"),
  Time = c(pc.time[3], hc.time[3], gs.time[3])
)

time_plot <- ggplot(time_data, aes(x = Algorithm, y = Time, fill = Algorithm)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = round(Time, 4)), vjust = -0.3, size = 3.5) +
  labs(title = "Computational Cost",
       x = "Algorithm", y = "Time (seconds)") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.6, size = 12, face = "bold"),
        axis.title.x = element_blank()) + 
  scale_y_continuous(expand = c(0,0),limits = c(0,0.4)) +
  scale_color_npg(alpha = 0.8) + scale_fill_npg(alpha = 0.8)

time_plot

# evaluate by 4 metrics
evaluate_model <- function(true_dag, pred_dag) {
  tp <- sum((true_dag > 0) & (pred_dag == 1))
  tn <- sum((true_dag == 0) & (pred_dag == 0))
  fp <- sum((true_dag == 0) & (pred_dag == 1))
  fn <- sum((true_dag > 0) & (pred_dag == 0))
  
  precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0
  recall <- if ((tp + fn) > 0) tp / (tp + fn) else 0
  f1 <- if ((precision + recall) > 0) 2 * (precision * recall) / (precision + recall) else 0
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  
  return(c(precision, recall, f1, accuracy))
}

# generate matrics dataframe
metrics <- data.frame(
  Method = rep(c("PC Algorithm", "Hill-Climbing", "Greedy Search"), each = 4),
  Metric = rep(c("Precision", "Recall", "F1 Score", "Accuracy"), 3),
  Value = c(pc_metrics, hc_metrics, gs_metrics)
)

# visualization
optimize_plot <- function(p) {
  p + scale_color_npg(alpha = 0.8)+scale_fill_npg(alpha = 0.8) +  
    theme_classic() +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_text(aes(label = round(Value, 3)), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5) + 
    theme(plot.title = element_text(hjust = 0.5),
          axis.title.x = element_blank(),
          panel.grid=element_blank()
    ) +
    scale_y_continuous(expand = c(0,0),limits = c(0,1.05)) 
  }


p1 <- ggplot(metrics[metrics$Metric == "Precision", ],
             aes(x = Method, y = Value, fill = Method)) +
  labs(title = "Precision", y = "Precision")

p2 <- ggplot(metrics[metrics$Metric == "Recall", ],
             aes(x = Method, y = Value, fill = Method)) +
  labs(title = "Recall", y = "Recall")

p3 <- ggplot(metrics[metrics$Metric == "F1 Score", ],
             aes(x = Method, y = Value, fill = Method)) +
  labs(title = "F1 Score", y = "F1 Score")

p4 <- ggplot(metrics[metrics$Metric == "Accuracy", ],
             aes(x = Method, y = Value, fill = Method)) +
  labs(title = "Accuracy", y = "Accuracy")


p1 <- optimize_plot(p1)
p2 <- optimize_plot(p2)
p3 <- optimize_plot(p3)
p4 <- optimize_plot(p4)


# concatenate all plots
cowplot::plot_grid(p4,p1,p2,p3,
  labels = c("A", "B", "C", "D"),
  ncol = 2)


# network visualization
plot_network <- function(adj_matrix, title) {
  graph <- graph_from_adjacency_matrix(adj_matrix, mode = "directed", diag = FALSE)
  plot(graph, main = title, vertex.size = 20, vertex.label.cex = 2, edge.arrow.size = 1)
}

# ground truth
adj_matrix2 = adj_matrix
adj_matrix2[adj_matrix2 > 0] = 1
adj_matrix2
par(mfrow = c(1, 1))
plot_network(adj_matrix2, "DAG (Ground Truth)")

# inferred
par(mfrow = c(1, 1))
plot_network(pc_adj, "PC Algorithm Inferred DAG")
plot_network(hc_adj, "Hill-Climbing Inferred DAG")
plot_network(gs_adj, "Greedy Search Inferred DAG")

