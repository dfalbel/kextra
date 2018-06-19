as_nullable_integer <- function(x) {

  if (is.null(x))
    return(NULL)

  as.integer(x)
}


#' @importFrom tensorflow tf
tensorflow::tf
