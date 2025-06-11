import numpy as np
import itertools
import warnings
from scipy.special import comb


class SparseGrid:
    def __init__(self, n, d, varargin=None):
        self.n = n
        self.d = d
        self.varargin = varargin
        self.eta = n
        self.design = None
        self.labels = None
        self.grids = None
        self.coeff = None
        self.levelsets = None
        self.griddim = None
        self.generate()

    def generate(self):
        n = self.n
        d = self.d
        varargin = self.varargin
        if varargin is not None:
            X_in = varargin[0]
            # Transpose and convert to list of lists (MATLAB: X = varargin{1}')
            X = [[list(col) for col in zip(*X_in)] for _ in range(d)]
            X_temp = X.copy()
            X_tot = [[None] * d for _ in range(len(X[0]))]
            X_dim = np.zeros((len(X[0]), d), dtype=int)
            X_dim_tot = np.zeros((len(X[0]), d), dtype=int)

            for k in range(d):  # MATLAB: for k = 1:d
                for j in range(len(X[0])):  # MATLAB: for j = 1:length(X)
                    if j > 0:  # MATLAB: if j > 1
                        # MATLAB: X{j,k} = setdiff(X_temp{j,k},X_temp{j-1,k})
                        X[j][k] = list(set(X_temp[j][k]) - set(X_temp[j - 1][k]))
                    # MATLAB: X_tot{j,k} = sort([X{1:j,k}])
                    all_vals = sorted(set().union(*[X[i][k] for i in range(j + 1)]))
                    X_tot[j][k] = all_vals
                    # MATLAB: X_dim(j,k) = length(X{j,k})
                    X_dim[j, k] = len(X[j][k])
                    # MATLAB: X_dim_tot(j,k) = length(X_tot{j,k})
                    X_dim_tot[j, k] = len(X_tot[j][k])
        else:
            X = [[[] for _ in range(d)] for _ in range(n - d + 1)]
            X_tot = [[[] for _ in range(d)] for _ in range(n - d + 1)]
            X_dim = np.zeros((n - d + 1, d), dtype=int)
            X_dim_tot = np.zeros((n - d + 1, d), dtype=int)
            for k in range(d):  # MATLAB: for k = 1:d
                for j in range(n - d + 1):  # MATLAB: for j = 1:n-d+1
                    # MATLAB: X{j,k}=1/2^j:1/2^(j-1):1-1/2^j
                    step = 1 / (2 ** (j + 1))
                    start = step
                    stop = 1 - step + 1e-10  # Avoid floating point issues
                    X[j][k] = np.arange(start, stop, 2 * step).tolist()
                for j in range(n - d + 1):
                    # MATLAB: X_dim(j,k) = length(X{j})
                    X_dim[j, k] = len(X[j][k])
                    # MATLAB: X_tot{j,k} = sort([X{1:j}])
                    all_vals = sorted(set().union(*[X[i][k] for i in range(j + 1)]))
                    X_tot[j][k] = all_vals
                    # MATLAB: X_dim_tot(j,k) = length(X_tot{j})
                    X_dim_tot[j, k] = len(X_tot[j][k])

        # Calculate totlevelsets and numlevelset (MATLAB: totlevelsets = 0; ...)
        totlevelsets = 0
        numlevelset = np.zeros((n - d + 1, 2), dtype=int)
        for i in range(d, n + 1):  # MATLAB: for i = d:n
            # MATLAB: totlevelsets = totlevelsets+nchoosek(i-1,d-1)
            binom = comb(i - 1, d - 1, exact=True)
            totlevelsets += binom
            # MATLAB: numlevelset(i-d+1,:) = [totlevelsets-binom+1, totlevelsets]
            numlevelset[i - d] = [totlevelsets - binom + 1, totlevelsets]

        # MATLAB: totlevelset =ones(totlevelsets,d)
        totlevelset = np.ones((totlevelsets, d), dtype=int)
        k = 0
        for i in range(d, n + 1):  # MATLAB: for i = d:n
            # MATLAB: totlevelset(k+1:k+nchoosek(i-1,d-1),:) = spgetseq((i-d),d)+1
            binom = comb(i - 1, d - 1, exact=True)
            seq = spgetseq(i - d, d) + 1
            totlevelset[k:k + binom] = seq
            k += binom

        # Calculate coefficients (MATLAB: coeff = zeros(size(totlevelset,1),1); ...)
        coeff = np.zeros(totlevelset.shape[0], dtype=int)
        for j in range(totlevelset.shape[0]):  # MATLAB: for j = 1:size(totlevelset,1)
            abs_j = np.sum(totlevelset[j, :])  # MATLAB: abs_j = sum(totlevelset(j,:))
            if max(d, n - d + 1) <= abs_j:  # MATLAB: if max(d,n-d+1)<= abs_j
                # MATLAB: coeff(j) = (-1)^(n-abs_j)*nchoosek(d-1,n-abs_j)
                sign = (-1) ** (n - abs_j)
                binom_val = int(comb(d - 1, n - abs_j, exact=True))
                coeff[j] = sign * binom_val

        # Calculate total points (MATLAB: num_points = 0; ...)
        num_points = 0
        for j in range(totlevelset.shape[0]):  # MATLAB: for j = 1:size(totlevelset,1)
            # MATLAB: q_inds = sub2ind(size(X_dim), totlevelset(j,:), 1:d)
            # Instead of sub2ind, use direct indexing with offset
            q_inds = [totlevelset[j, dim] - 1 for dim in range(d)]  # Adjust for 0-index
            num_points += np.prod([X_dim[i, dim] for dim, i in enumerate(q_inds)])

        # Initialize points array (MATLAB: X_points = zeros(num_points,d))
        X_points = np.zeros((num_points, d))

        # Build C_mat (MATLAB: C_mat = zeros(n,d,n-d); ...)
        C_mat = np.zeros((n, d, n - d), dtype=int)
        for k_val in range(1, n + 1):  # MATLAB: for k= 1:n
            for i in range(d):  # MATLAB: for i = 1:d
                for j in range(1, n - d + 1):  # MATLAB: for j = 1:(n-d)
                    # MATLAB: C_mat(k,i,j) = counting(k,i,j)
                    C_mat[k_val - 1, i, j - 1] = int(counting(k_val, i + 1, j))

        # Initialize variables for point generation
        k_index = 0  # MATLAB: k=0
        label = [None] * totlevelset.shape[0]  # MATLAB: label = cell(...)
        ulabel = [None] * totlevelset.shape[0]
        ind_store2 = [None] * totlevelset.shape[0]

        # Temporary storage (MATLAB: X_points_temp = zeros(max(...),d))
        max_dim = np.max(X_dim[n - d, :]) if n - d < X_dim.shape[0] else 1
        X_points_temp = np.zeros((max_dim, d))

        # Main loop through level sets (MATLAB: for j = 1:size(totlevelset,1))
        for j in range(totlevelset.shape[0]):
            # Find level set index (MATLAB: for l = 1:(n-d+1) ... break; end)
            l_index = 0
            for l in range(numlevelset.shape[0]):
                if numlevelset[l, 0] <= j + 1 <= numlevelset[l, 1]:
                    l_index = l
                    break

            # Get dimensions (MATLAB: q_inds = sub2ind(...))
            q_inds = [totlevelset[j, dim] - 1 for dim in range(d)]  # 0-index adjustment
            q = [X_dim[q_inds[dim], dim] for dim in range(d)]

            # Generate grid points
            # MATLAB: X_points_temp(1:prod(q),d) = repmat(...)
            dim_vals = []
            for dim in range(d):
                level = totlevelset[j, dim] - 1  # Convert to 0-index
                dim_vals.append(X[level][dim])  # X{j,dim} points

            # Create full factorial design
            grid = np.array(list(itertools.product(*dim_vals)))
            num_pts = grid.shape[0]

            # Store points (MATLAB: X_points(k+1:k+prod(q),:) = ...)
            X_points[k_index:k_index + num_pts] = grid
            ulabel[j] = list(range(k_index, k_index + num_pts))

            # Initialize label array (MATLAB: label{j} = zeros(prod(f),1))
            f = [X_dim_tot[q_inds[dim], dim] for dim in range(d)]
            label[j] = []

            # Handle hierarchical structure (MATLAB: if l>1 ...)
            if l_index > 0:  # l_index is 0-index (MATLAB l>1)
                # MATLAB: [~,vals] = find(totlevelset(j,:)>=2)
                vals = [dim for dim in range(d) if totlevelset[j, dim] >= 2]
                indexed_vals = []

                if l_index > 1:  # MATLAB: if l>2
                    for inter in range(len(vals)):
                        # Create modified level set
                        samp_level_set = totlevelset[j, :].copy()
                        samp_level_set[vals[inter]] -= 1
                        # MATLAB: indexed_vals(inter) = counter_f(...)
                        idx_val = counter_f(samp_level_set, C_mat) + numlevelset[l_index - 1, 0] - 1
                        indexed_vals.append(idx_val - 1)

                    # MATLAB: ind_store2{j} = sort(unique(...))
                    # ind_store2[j] = sorted(set(indexed_vals))
                    ind_store2[j] = sorted(merge_and_dedup(ind_store2, indexed_vals) + indexed_vals)
                else:
                    indexed_vals = [0]  # First level set
                    ind_store2[j] = indexed_vals

                # Assemble labels from lower levels
                for j0 in ind_store2[j]:
                    label[j].extend(ulabel[j0])

            # Add current level points (MATLAB: label{j}(k_0+1:end) = ulabel{j})
            label[j].extend(ulabel[j])

            # Sort points (MATLAB: [~,ind_val] = sortrows(...))
            points_to_sort = X_points[label[j]]
            sorted_indices = np.lexsort(points_to_sort[:, ::-1].T)
            label[j] = [label[j][i] for i in sorted_indices]

            # Update point counter
            k_index += num_pts

        self.design = X_points
        self.labels = label
        self.grids = X_tot
        self.coeff = coeff
        self.levelsets = totlevelset - 1
        self.griddim = X_dim_tot


def counting(n, d, c):
    """
    Helper function for index calculation
    MATLAB: function index_vals_temp = counting(n,d,c)
    """
    # MATLAB: warning off
    warnings.filterwarnings('ignore')
    index_vals_temp = 0
    # MATLAB: if d-2>=0
    if d - 2 >= 0:
        # MATLAB: if n-c>=d-2
        if n - c >= d - 2:
            # MATLAB: for lcv = 2:1:c
            for lcv in range(2, c + 1):
                # MATLAB: index_vals_temp += nchoosek(n-lcv, d-2)
                if n - lcv >= d - 2:
                    index_vals_temp += comb(n - lcv, d - 2, exact=True)
    # MATLAB: warning on
    warnings.resetwarnings()
    return index_vals_temp


def counter_f(vec_val, C_mat):
    """
    Index calculation for level sets
    MATLAB: function k = counter_f(vec_val, C_mat)
    """
    n_val = np.sum(vec_val)  # MATLAB: n = sum(vec_val)
    d_val = len(vec_val)  # MATLAB: d = length(vec_val)
    k = 1  # MATLAB: k=1

    # MATLAB: for i = d:-1:2
    for i in range(d_val - 1, 0, -1):  # i from d-1 down to 1 (0-index)
        # MATLAB: if vec_val(i)>1
        if vec_val[i] > 1:
            # Calculate sum of higher dimensions (MATLAB: n-sum(vec_val(i+1:end))
            sum_higher = np.sum(vec_val[i + 1:])
            # MATLAB: k = k + C_mat(n-sum, i, vec_val(i))
            # Adjust indices for 0-based and C_mat dimensions
            n_idx = n_val - sum_higher - 1
            d_idx = i
            c_idx = vec_val[i] - 1
            k += C_mat[int(n_idx), int(d_idx), int(c_idx)]
    return k


def merge_and_dedup(x, y):
    merged_list = []
    for idx in y:
        if idx < len(x):
            merged_list.extend(x[idx])

    seen = set()
    result = []
    for item in merged_list:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def spgetseq(n, d):
    nlevels = int(comb(n + d - 1, d - 1))
    seq = np.zeros([nlevels, d])
    seq[0, 0] = n
    maxn = n
    for k in range(2, nlevels + 1):
        if seq[k - 2, 0] > 0:
            seq[k - 1, 0] = seq[k - 2, 0] - 1
            for l in range(2, d + 1):
                if seq[k - 2, l - 1] < maxn:
                    seq[k - 1, l - 1] = seq[k - 2, l - 1] + 1
                    for m in range(l + 1, d + 1):
                        seq[k - 1, m - 1] = seq[k - 2, m - 1]
                    break
        else:
            s = 0
            for l in range(2, d + 1):
                if seq[k - 2, l - 1] < maxn:
                    seq[k - 1, l - 1] = seq[k - 2, l - 1] + 1
                    s += seq[k - 1, l - 1]
                    for m in range(l + 1, d + 1):
                        seq[k - 1, m - 1] = seq[k - 2, m - 1]
                        s += seq[k - 1, m - 1]
                    break
                else:
                    temp = 0
                    for m in range(l + 2, d + 1):
                        temp += seq[k - 2, m - 1]
                    maxn = n - temp
                    seq[k - 1, l - 1] = 0
            seq[k - 1, 0] = n - s
            maxn = n - s
    return seq
