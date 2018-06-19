library(keras)
library(kextra)
library(tfdatasets)


# Download and extract data -----------------------------------------------

# dir.create("data")
#
# download.file(
#   url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
#   destfile = "data/speech_commands_v0.01.tar.gz"
# )
#
# untar("data/speech_commands_v0.01.tar.gz", exdir = "data/speech_commands_v0.01")


# Read files --------------------------------------------------------------

library(stringr)
library(dplyr)

files <- fs::dir_ls(
  path = "../speech-keras/data/speech_commands_v0.01/",
  recursive = TRUE,
  glob = "*.wav"
)

files <- files[!str_detect(files, "background_noise")]

df <- data_frame(
  fname = files,
  class = fname %>% str_extract("1/.*/") %>%
    str_replace_all("1/", "") %>%
    str_replace_all("/", ""),
  class_id = class %>% as.factor() %>% as.integer() - 1L
)

# Create a generator ------------------------------------------------------

audio_ops <- tf$contrib$framework$python$ops$audio_ops

data_generator <- function(df, batch_size, shuffle = TRUE) {

  ds <- tensor_slices_dataset(df)

  if (shuffle)
    ds <- ds %>% dataset_shuffle(buffer_size = 5000)

  ds <- ds %>%
    dataset_map(function(obs) {
      # decoding wav files
      audio_binary <- tf$read_file(tf$reshape(obs$fname, shape = list()))
      wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)


      # transform the class_id into a one-hot encoded vector
      response <- tf$one_hot(obs$class_id, 30L)

      list(wav$audio, response)
    }) %>%
    dataset_repeat()

  ds <- ds %>%
    dataset_padded_batch(32, list(list(16000, 1), list(NULL)))

  ds
}

id_train <- sample.int(nrow(df), 0.7*nrow(df))
ds_train <- data_generator(df[id_train,], 32)
ds_valid <- data_generator(df[-id_train,], 32, shuffle = FALSE)

# Model definition --------------------------------------------------------

input <- layer_input(shape = c(16000L, 1L))
spectrogram <- layer_spectrogram(input, 320L, 160L, log_compress = TRUE, log_offset = 0.01)
output <- spectrogram %>%
  layer_conv_2d(filters = 32, kernel_size =  c(3,3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64,kernel_size =  c(3,3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_flatten() %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(30, activation = "softmax")

model <- keras_model(input, output)

model %>%
  compile(
    optimizer = "adam", loss = "categorical_crossentropy", metrics = "accuracy"
  )

model %>% fit_generator(
  ds_train,
  steps_per_epoch = 0.7*nrow(df),
  epochs = 10,
  validation_data = ds_valid,
  validation_steps = 0.3*nrow(df)
  )



