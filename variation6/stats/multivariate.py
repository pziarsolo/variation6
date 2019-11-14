
from numpy import dot
from numpy.linalg import svd
import variation6.array as va


def _center_matrix(matrix):
    'It centers the matrix'
    means = matrix.mean(axis=0)
    return matrix - means


def do_pca(variations):
    'It does a Principal Component Analysis'
    # transform the genotype data into a 2-dimensional matrix where each cell
    # has the number of non-reference alleles per call
    gts012 = va.gts_as_mat012(variations)
    matrix = gts012.T
    va.make_sure_array_is_in_memory(matrix)

    n_rows, n_cols = matrix.shape
    if n_cols < n_rows:
        # This restriction is in the matplotlib implementation, but I don't
        # know the reason
        msg = 'The implementation requires more SNPs than samples'
        raise RuntimeError(msg)

    # Implementation based on the matplotlib PCA class
    cen_matrix = _center_matrix(matrix)
    # The following line should be added from a example to get the correct
    # variances
    # cen_scaled_matrix = cen_matrix / math.sqrt(n_rows - 1)
    cen_scaled_matrix = cen_matrix

    singular_vals, princomps = svd(cen_scaled_matrix, full_matrices=False)[1:]
    eig_vals = singular_vals ** 2
    pcnts = eig_vals / eig_vals.sum() * 100.0
    projections = dot(princomps, cen_matrix.T).T

    return {'projections': projections,
            'var_percentages': pcnts,
            'princomps': princomps}
