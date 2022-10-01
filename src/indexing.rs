/// Given an index into a flattened 2-d matrix, find the 2-d coordinate of that index
///
/// `nmajor` is the length of the major axis of the matrix (i.e. ncols for column-major, nrows for row-major)
///
/// The returned index is in (minor, major) axis order
#[inline(always)]
pub fn index_to_2d(index: usize, nmajor: usize) -> (usize, usize) {
    let i = index / nmajor;
    let j = index - (i * nmajor);
    (i, j)
}

/// Given a minor axis index and the size of the major axis,
/// return the start and end slice positions to slice along the major axis at the index
#[inline(always)]
pub fn slice_indices(index: usize, nmajor: usize) -> (usize, usize) {
    let xs = index * nmajor;
    let xe = xs + nmajor;
    (xs, xe)
}

#[cfg(test)]
mod test_index_to_2d {
    use super::index_to_2d;

    #[test]
    fn test_index_tl() {
        /*
          x o o
          o o o
          o o o
          o o o
          o o o
        */
        assert_eq!(index_to_2d(0, 5), (0, 0));
    }

    #[test]
    fn test_index_tr() {
        /*
          o o x
          o o o
          o o o
          o o o
          o o o
        */
        assert_eq!(index_to_2d(10, 5), (2, 0));
    }

    #[test]
    fn test_index_bl() {
        /*
          o o o
          o o o
          o o o
          o o o
          x o o
        */
        assert_eq!(index_to_2d(4, 5), (0, 4));
    }

    #[test]
    fn test_index_br() {
        /*
          o o o
          o o o
          o o o
          o o o
          o o x
        */
        assert_eq!(index_to_2d(14, 5), (2, 4));
    }

    #[test]
    fn test_index_middle() {
        /*
          o o o
          o o o
          o x o
          o o o
          o o o
        */
        assert_eq!(index_to_2d(7, 5), (1, 2));
    }
}

#[cfg(test)]
mod test_slice_indices {
    use super::slice_indices;

    #[test]
    fn test_first() {
        /*
          x o o o o
          x o o o o
          x o o o o
        */
        assert_eq!(slice_indices(0, 3), (0, 3));
    }

    #[test]
    fn test_last() {
        /*
          o o o o x
          o o o o x
          o o o o x
        */
        assert_eq!(slice_indices(4, 3), (12, 15));
    }

    #[test]
    fn test_middle() {
        /*
          o o x o o
          o o x o o
          o o x o o
        */
        assert_eq!(slice_indices(2, 3), (6, 9))
    }
}
