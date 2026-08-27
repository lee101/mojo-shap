"""Exact path-dependent TreeSHAP and KernelSHAP masking kernels."""

from max.algorithm import parallelize
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of


comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def extend_path(
    feature: IPtr,
    zero_fraction: FPtr,
    one_fraction: FPtr,
    pweight: FPtr,
    depth: Int,
    zero: Float64,
    one: Float64,
    split_feature: Int64,
):
    feature[depth] = split_feature
    zero_fraction[depth] = zero
    one_fraction[depth] = one
    if depth == 0:
        pweight[0] = 1.0
    else:
        pweight[depth] = 0.0
    for offset in range(depth):
        var i = depth - 1 - offset
        pweight[i + 1] += (
            one * pweight[i] * Float64(i + 1) / Float64(depth + 1)
        )
        pweight[i] = (
            zero * pweight[i] * Float64(depth - i) / Float64(depth + 1)
        )


def unwind_path(
    feature: IPtr,
    zero_fraction: FPtr,
    one_fraction: FPtr,
    pweight: FPtr,
    depth: Int,
    path_index: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var one = one_fraction[path_index]
    var zero = zero_fraction[path_index]
    var next_one = pweight[depth]
    for offset in range(depth):
        var i = depth - 1 - offset
        if one != 0.0:
            var tmp = pweight[i]
            pweight[i] = (
                next_one * Float64(depth + 1) / (Float64(i + 1) * one)
            )
            next_one = tmp - (
                pweight[i] * zero * Float64(depth - i) / Float64(depth + 1)
            )
        elif zero != 0.0:
            pweight[i] = (
                pweight[i] * Float64(depth + 1)
                / (zero * Float64(depth - i))
            )
    var copy_count = depth - path_index
    var vector_end = path_index + copy_count - copy_count % W
    for i in range(path_index, vector_end, W):
        feature.store(i, feature.load[width=W](i + 1))
        zero_fraction.store(i, zero_fraction.load[width=W](i + 1))
        one_fraction.store(i, one_fraction.load[width=W](i + 1))
    for i in range(vector_end, depth):
        feature[i] = feature[i + 1]
        zero_fraction[i] = zero_fraction[i + 1]
        one_fraction[i] = one_fraction[i + 1]


def unwound_path_sum(
    zero_fraction: FPtr,
    one_fraction: FPtr,
    pweight: FPtr,
    depth: Int,
    path_index: Int,
) -> Float64:
    var one = one_fraction[path_index]
    var zero = zero_fraction[path_index]
    var next_one = pweight[depth]
    var total = 0.0
    for offset in range(depth):
        var i = depth - 1 - offset
        if one != 0.0:
            var tmp = (
                next_one * Float64(depth + 1) / (Float64(i + 1) * one)
            )
            total += tmp
            next_one = pweight[i] - (
                tmp * zero * Float64(depth - i) / Float64(depth + 1)
            )
        elif zero != 0.0:
            total += (
                pweight[i] * Float64(depth + 1)
                / (zero * Float64(depth - i))
            )
    return total


def recurse_tree(
    x: FPtr,
    n_features: Int,
    children_left: IPtr,
    children_right: IPtr,
    split_feature: IPtr,
    threshold: FPtr,
    value: FPtr,
    node_weight: FPtr,
    missing_left: IPtr,
    n_outputs: Int,
    phi: FPtr,
    node: Int,
    depth: Int,
    parent_feature: IPtr,
    parent_zero: FPtr,
    parent_one: FPtr,
    parent_weight: FPtr,
    incoming_zero: Float64,
    incoming_one: Float64,
    incoming_feature: Int64,
    tree_scale: Float64,
):
    var path_feature = parent_feature + depth + 1
    var path_zero = parent_zero + depth + 1
    var path_one = parent_one + depth + 1
    var path_weight = parent_weight + depth + 1
    comptime W = simd_width_of[DType.float64]()
    var path_count = depth + 1
    var vector_end = path_count - path_count % W
    for i in range(0, vector_end, W):
        path_feature.store(i, parent_feature.load[width=W](i))
        path_zero.store(i, parent_zero.load[width=W](i))
        path_one.store(i, parent_one.load[width=W](i))
        path_weight.store(i, parent_weight.load[width=W](i))
    for i in range(vector_end, path_count):
        path_feature[i] = parent_feature[i]
        path_zero[i] = parent_zero[i]
        path_one[i] = parent_one[i]
        path_weight[i] = parent_weight[i]

    extend_path(
        path_feature,
        path_zero,
        path_one,
        path_weight,
        depth,
        incoming_zero,
        incoming_one,
        incoming_feature,
    )

    if children_right[node] < 0:
        for i in range(1, depth + 1):
            var w = unwound_path_sum(
                path_zero, path_one, path_weight, depth, i
            )
            var f = Int(path_feature[i])
            var factor = (
                w * (path_one[i] - path_zero[i]) * tree_scale
            )
            var output_end = n_outputs - n_outputs % W
            var phi_base = phi + f * n_outputs
            var value_base = value + node * n_outputs
            for o in range(0, output_end, W):
                phi_base.store(
                    o,
                    phi_base.load[width=W](o)
                    + factor * value_base.load[width=W](o),
                )
            for o in range(output_end, n_outputs):
                phi[f * n_outputs + o] += factor * value[node * n_outputs + o]
        return

    var f = Int(split_feature[node])
    var hot = Int(children_left[node])
    if x[f] != x[f]:
        if missing_left[node] == 0:
            hot = Int(children_right[node])
    elif x[f] > threshold[node]:
        hot = Int(children_right[node])
    var cold = Int(children_right[node])
    if hot == cold:
        cold = Int(children_left[node])

    var hot_zero = 0.0
    var cold_zero = 0.0
    if node_weight[node] != 0.0:
        hot_zero = node_weight[hot] / node_weight[node]
        cold_zero = node_weight[cold] / node_weight[node]

    var path_index = 0
    for i in range(depth + 1):
        if path_feature[i] == Int64(f):
            path_index = i
            break

    var split_zero = 1.0
    var split_one = 1.0
    var child_depth = depth
    if path_index != 0:
        split_zero = path_zero[path_index]
        split_one = path_one[path_index]
        unwind_path(
            path_feature, path_zero, path_one, path_weight, depth, path_index
        )
        child_depth -= 1

    recurse_tree(
        x,
        n_features,
        children_left,
        children_right,
        split_feature,
        threshold,
        value,
        node_weight,
        missing_left,
        n_outputs,
        phi,
        hot,
        child_depth + 1,
        path_feature,
        path_zero,
        path_one,
        path_weight,
        hot_zero * split_zero,
        split_one,
        Int64(f),
        tree_scale,
    )
    recurse_tree(
        x,
        n_features,
        children_left,
        children_right,
        split_feature,
        threshold,
        value,
        node_weight,
        missing_left,
        n_outputs,
        phi,
        cold,
        child_depth + 1,
        path_feature,
        path_zero,
        path_one,
        path_weight,
        cold_zero * split_zero,
        0.0,
        Int64(f),
        tree_scale,
    )


def explain_tree_row(
    r: Int,
    x: FPtr,
    n_features: Int,
    children_left: IPtr,
    children_right: IPtr,
    split_feature: IPtr,
    threshold: FPtr,
    value: FPtr,
    node_weight: FPtr,
    missing_left: IPtr,
    tree_offsets: IPtr,
    tree_scale: FPtr,
    n_trees: Int,
    n_outputs: Int,
    phi: FPtr,
    path_feature: IPtr,
    path_zero: FPtr,
    path_one: FPtr,
    path_weight: FPtr,
):
    var row_size = n_features * n_outputs
    var row_phi = phi + r * row_size
    comptime W = simd_width_of[DType.float64]()
    var vector_end = row_size - row_size % W
    for i in range(0, vector_end, W):
        row_phi.store(i, SIMD[DType.float64, W](0.0))
    for i in range(vector_end, row_size):
        row_phi[i] = 0.0
    for t in range(n_trees):
        var offset = Int(tree_offsets[t])
        recurse_tree(
            x + r * n_features,
            n_features,
            children_left + offset,
            children_right + offset,
            split_feature + offset,
            threshold + offset,
            value + offset * n_outputs,
            node_weight + offset,
            missing_left + offset,
            n_outputs,
            row_phi,
            0,
            0,
            path_feature,
            path_zero,
            path_one,
            path_weight,
            1.0,
            1.0,
            -1,
            tree_scale[t],
        )


@export("msh_tree_shap")
def tree_shap(
    x_addr: Int,
    n_rows: Int,
    n_features: Int,
    children_left_addr: Int,
    children_right_addr: Int,
    split_feature_addr: Int,
    threshold_addr: Int,
    value_addr: Int,
    node_weight_addr: Int,
    missing_left_addr: Int,
    tree_offsets_addr: Int,
    tree_scale_addr: Int,
    n_trees: Int,
    n_outputs: Int,
    phi_addr: Int,
    path_feature_addr: Int,
    path_zero_addr: Int,
    path_one_addr: Int,
    path_weight_addr: Int,
    path_stride: Int,
) abi("C") -> Int64:
    if (
        x_addr == 0 or children_left_addr == 0 or children_right_addr == 0
        or split_feature_addr == 0 or threshold_addr == 0 or value_addr == 0
        or node_weight_addr == 0 or missing_left_addr == 0
        or tree_offsets_addr == 0 or tree_scale_addr == 0 or phi_addr == 0
        or path_feature_addr == 0 or path_zero_addr == 0 or path_one_addr == 0
        or path_weight_addr == 0
    ):
        return 1
    if n_rows <= 0 or n_features <= 0 or n_trees <= 0 or n_outputs <= 0:
        return 2
    if path_stride < 0:
        return 3
    var x = FPtr(unsafe_from_address=x_addr)
    var children_left = IPtr(unsafe_from_address=children_left_addr)
    var children_right = IPtr(unsafe_from_address=children_right_addr)
    var split_feature = IPtr(unsafe_from_address=split_feature_addr)
    var threshold = FPtr(unsafe_from_address=threshold_addr)
    var value = FPtr(unsafe_from_address=value_addr)
    var node_weight = FPtr(unsafe_from_address=node_weight_addr)
    var missing_left = IPtr(unsafe_from_address=missing_left_addr)
    var tree_offsets = IPtr(unsafe_from_address=tree_offsets_addr)
    var tree_scale = FPtr(unsafe_from_address=tree_scale_addr)
    var phi = FPtr(unsafe_from_address=phi_addr)
    var path_feature = IPtr(unsafe_from_address=path_feature_addr)
    var path_zero = FPtr(unsafe_from_address=path_zero_addr)
    var path_one = FPtr(unsafe_from_address=path_one_addr)
    var path_weight = FPtr(unsafe_from_address=path_weight_addr)

    @__parameter
    def process_row(r: Int, scratch_row: Int):
        var scratch_offset = scratch_row * path_stride
        explain_tree_row(
            r,
            x,
            n_features,
            children_left,
            children_right,
            split_feature,
            threshold,
            value,
            node_weight,
            missing_left,
            tree_offsets,
            tree_scale,
            n_trees,
            n_outputs,
            phi,
            path_feature + scratch_offset,
            path_zero + scratch_offset,
            path_one + scratch_offset,
            path_weight + scratch_offset,
        )

    if path_stride == 0:
        for r in range(n_rows):
            process_row(r, 0)
        return 0

    var tasks = min(n_rows, 64)
    var rows_per_task = (n_rows + tasks - 1) // tasks

    @always_inline
    def process_task(task: Int) {imm n_rows, imm rows_per_task}:
        var start = task * rows_per_task
        var end = min(start + rows_per_task, n_rows)
        for r in range(start, end):
            process_row(r, task)

    initialize_runtime()
    parallelize(process_task, tasks, tasks)
    return 0


@export("msh_make_synthetic")
def make_synthetic(
    x_addr: Int,
    background_addr: Int,
    masks_addr: Int,
    varying_addr: Int,
    synthetic_addr: Int,
    n_background: Int,
    n_features: Int,
    n_masks: Int,
    n_varying: Int,
) abi("C") -> Int64:
    if (
        x_addr == 0 or background_addr == 0 or masks_addr == 0
        or varying_addr == 0 or synthetic_addr == 0
    ):
        return 1
    if n_background <= 0 or n_features <= 0 or n_masks <= 0 or n_varying <= 0:
        return 2
    var x = FPtr(unsafe_from_address=x_addr)
    var background = FPtr(unsafe_from_address=background_addr)
    var masks = FPtr(unsafe_from_address=masks_addr)
    var varying = IPtr(unsafe_from_address=varying_addr)
    var synthetic = FPtr(unsafe_from_address=synthetic_addr)
    for mask_index in range(n_masks):
        for b in range(n_background):
            var dst = synthetic + (mask_index * n_background + b) * n_features
            var src = background + b * n_features
            for j in range(n_features):
                dst[j] = src[j]
            for j in range(n_varying):
                if masks[mask_index * n_varying + j] != 0.0:
                    var f = Int(varying[j])
                    dst[f] = x[f]
    return 0
