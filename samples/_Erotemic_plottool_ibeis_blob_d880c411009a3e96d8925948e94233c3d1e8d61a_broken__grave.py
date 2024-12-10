

def test_integral_label_colormap():
    """
    UNFINISHED

    Above 0 use a inverted hot scale and less than that use special colors

    References:
        http://stackoverflow.com/questions/18704353/correcting-matplotlib-colorbar-ticks
        http://stackoverflow.com/questions/15908371/matplotlib-colorbars-and-its-text-labels
        http://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar

    Example:
        >>> from plottool.draw_func2 import *  # NOQA
    """

    def label_domain(unique_scalars):
        diff = np.diff(unique_scalars)
        # Find the holes in unique_scalars
        missing_vals = []
        for diffx in np.where(diff > 1)[0]:
            missing_vals.extend([(unique_scalars[diffx] + x + 1) for x in range(diff[diffx] - 1)])

        # Find the indicies of those holes
        missing_ixs = np.array(missing_vals) - min_
        assert all([val not in unique_scalars for val in missing_vals])

        domain = np.array([x for ix, x in enumerate(unique_scalars) if ix not in missing_ixs])
        domain -= min_
        return domain

    from plottool import df2
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    import utool

    fig, ax = plt.subplots()
    np.random.seed(0)
    data = (np.random.random((10, 10)) * 13).astype(np.int32) - 2
    data[data == 0] = 12

    unique_scalars = np.array(sorted(np.unique(data)))
    max_ = unique_scalars.max()
    min_ = unique_scalars.min()
    range_ = max_ - min_
    bounds = np.linspace(min_, max_ + 1, range_ + 2)

    base_colormap = df2.reverse_colormap(plt.get_cmap('hot'))
    # Get a few more colors than we actually need so we don't hit the bottom of
    # the cmap
    colors_ix = np.concatenate((np.linspace(0, 1., range_ + 2), (0., 0., 0., 0.)))
    colors_rgba = base_colormap(colors_ix)
    val2_special_rgba = {
        -1: df2.UNKNOWN_PURP,
        -2: df2.LIGHT_BLUE,
    }
    def get_new_color(ix, val):
        if val in val2_special_rgba:
            return val2_special_rgba[val]
        else:
            return colors_rgba[ix - len(val2_special_rgba) + 1]
    special_colors = [get_new_color(ix, val) for ix, val in enumerate(bounds)]

    cmap = mpl.colors.ListedColormap(special_colors)

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(data, interpolation='nearest', cmap=cmap, norm=norm)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    sm.set_clim(-.5, range_ + 0.5)
    colorbar = plt.colorbar(sm)

    missing_ixs = utool.find_nonconsec_indicies(unique_scalars, bounds)
    sel_bounds = np.array([x for ix, x in enumerate(bounds) if ix not in missing_ixs])

    ticks = sel_bounds + .5
    ticklabels = sel_bounds
    colorbar.set_ticks(ticks)  # tick locations
    colorbar.set_ticklabels(ticklabels)  # tick labels

