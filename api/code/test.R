# make a list of 3 numbers

library("here")

list_of_numbers <- list(1, 2, 3)

print(getwd())


setwd(here("api", "code", "12"))

print(getwd())
saveRDS(list_of_numbers, "test.RDS")
