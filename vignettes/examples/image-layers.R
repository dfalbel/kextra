library(keras)
library(kextra)

cifar10 <- dataset_cifar10()
cifar10$train$y <- to_categorical(cifar10$train$y)
cifar10$test$y <- to_categorical(cifar10$test$y)

x <- cifar10$train$x
y <- cifar10$train$y


input <- layer_input(c(32, 32, 3))
output <- input %>%
  layer_image_resize(size = c(20, 20)) %>%
  layer_image_rgb_to_grayscale() %>%
  layer_conv_2d(kernel_size = c(2,2), filters = 32, padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(2,2)) %>%
  layer_max_pooling_2d() %>%
  layer_flatten() %>%
  layer_dense(10)

model <- keras_model(input, output)

model %>%
  compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

model %>%
  fit(
    x = x, y = y,
    validation_split = 0.1
  )
