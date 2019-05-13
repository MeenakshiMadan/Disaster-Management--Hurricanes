## Prepared by Miao Zheng ##
library(ggplot2)
library(tidyverse)
library(micromapST)
library(scales)
library(plyr)

harvey <- read.csv(file='Project4.csv', header=TRUE)

## Scatterplot of polarity vs. subjectivity. Used position_jitter to prevent overplotting.
plot1 <- ggplot(data=harvey, aes(x=polarity, y=subjectivity)) + 
  geom_point(alpha=1/20,position=position_jitter(h=0)) +
  ggtitle("Polarity vs. Subjectivity") +
  theme(plot.title = element_text(hjust = 0.5))
plot1

## Separate the previous scatterplot into clusters using k-means 
plot2 <- ggplot(data=harvey, aes(x=polarity, y=subjectivity)) + 
  geom_point(alpha=1/20,position=position_jitter(h=0), 
             aes(colour= factor(kmeans(scale(cbind(polarity,subjectivity)), centers=3)$cluster))) +
  labs(title="Polarity vs. Subjectivity",
       color="Clusters") +
  theme(plot.title = element_text(hjust = 0.5)) 
plot2a<- plot2 + scale_color_brewer(palette="Dark2")

plot2a

## Frequency Distribution of Polarity
plot3 <- ggplot(data=harvey, aes(x=polarity)) +
  geom_histogram(color="darkblue", fill="lightblue", bins=10) +
  xlab("Polarities") + 
  ylab("Frequency") +
  ggtitle("Sentiment Analysis of Tweets on Polarity") +
  theme(plot.title = element_text(hjust = 0.5)) 
plot3

## Frequency Distribution of Subjectivity
plot4 <- ggplot(data=harvey, aes(x=subjectivity)) +
  geom_histogram(color="darkblue", fill="lightblue", bins=10) +
  xlab("Subjectivity") + 
  ylab("Frequency") +
  ggtitle("Sentiment Analysis of Tweets on Subjectivity") +
  theme(plot.title = element_text(hjust = 0.5)) 
plot4



