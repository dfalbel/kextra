
is_power_of_2 <- function(x) {

  if( x %% 2 != 0)
    return(FALSE)

  x <- x / 2

  while (x > 1) {
    x <- x / 2
  }

  if (x == 1)
    return(TRUE)
  else
    return(FALSE)

}

Spectrogram <- R6::R6Class(
  "Spectrogram",

  inherit = keras::KerasLayer,

  public = list(

    image_data_format = NULL,
    n_filter = NULL,
    n_hop = NULL,
    padding = NULL,
    power_spectrogram = NULL,
    return_decibel_spectrogram = NULL,
    trainable_kernel = NULL,

    output_dim = NULL,
    kernel = NULL,

    initialize = function(n_dft = 512, n_hop = NULL, padding = "same",
                          power_spectrogram = 2,
                          return_decibel_spectrogram = FALSE,
                          trainable_kernel = FALSE,
                          image_data_format = "default") {

      if (n_dft <= 1)
        stop("n_dft must be > 1, but n_dft=", n_dft)
      if (!is_power_of_2(n_dft))
        stop("n_dft must be power of 2, but n_dft=", n_dft)
      if (!is.logical(trainable_kernel))
        stop("trainable kernel must be logical but is ", class(trainable_kernel))
      if (!is.logical(return_decibel_spectrogram))
        stop("return_decibel_spectrogram must be logical but is ", class(return_decibel_spectrogram))
      if (!padding %in% c("same", "valid"))
        stop("padding must be in c('same', 'valid') but is ", padding)

      if (is.null(n_hop))
        n_hop <- n_dft %/% 2

      if (!image_data_format %in% c('default', 'channels_first', 'channels_last'))
        stop("image_data_format must be in c('default', 'channels_first', 'channels_last'), but is ", image_data_format)

      if(image_data_format == "default")
        self$image_data_format <- keras::k_image_data_format()
      else
        self$image_data_format <- image_data_format

      self$n_filter <- n_dft %/% 2 + 1
      self$trainable_kernel <- trainable_kernel
      self$n_hop <- n_hop
      self$padding <- padding
      self$power_spectrogram <- power_spectrogram
      self$return_decibel_spectrogram <- return_decibel_spectrogram
    },

    build = function(input_shape) {
      self$kernel <- self$add_weight(
        name = 'kernel',
        shape = list(input_shape[[2]], self$output_dim),
        initializer = initializer_random_normal(),
        trainable = TRUE
      )
    },

    call = function(x, mask = NULL) {
      k_dot(x, self$kernel)
    },

    compute_output_shape = function(input_shape) {
      list(input_shape[[1]], self$output_dim)
    }
  )
)
