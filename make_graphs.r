library(tidyverse)
library(data.table)








df <- (fread('analyzed/example_one_ana.csv', sep = ',', header = TRUE))

ggplot(df) + geom_line(aes(x = s, y = L, group = ID, color = as.factor(ID)))

df <- (fread('analyzed/fixing_from_high_ana.csv', sep = ',', header = TRUE))
ggplot(df) + geom_line(aes(x = s, y = L, group = ID, color = as.factor(ID)))

df <- (fread('analyzed/fixing_from_middle_ana.csv', sep = ',', header = TRUE))
ggplot(df) + geom_line(aes(x = s, y = L, group = ID, color = as.factor(ID)))

df <- (fread('analyzed/fixing_from_low_ana.csv', sep = ',', header = TRUE))
ggplot(df) + geom_line(aes(x = s, y = L, group = ID, color = as.factor(ID)))

