import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.style.use('ggplot')


def _get_mplot_fig_and_canvas(fhand, figsize=None):
    if fhand is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = Figure(figsize=figsize)

    canvas = FigureCanvas(fig)
    return fig, canvas


def _get_mplot_axes(axes, fhand, figsize=None, plot_type=111):
    if axes is not None:
        return axes, None, None

    fig, canvas = _get_mplot_fig_and_canvas(fhand, figsize=figsize)

    axes = fig.add_subplot(plot_type)
    return axes, canvas, fig


def _print_figure(canvas, fhand, no_interactive_win):
    if fhand is None:
        if not no_interactive_win:
            plt.show()
        return
    canvas.print_figure(fhand)


def plot_histogram(counts, edges, fhand=None, axes=None, vlines=None,
                   no_interactive_win=False, figsize=None,
                   mpl_params=None, bin_labels=None, **kwargs):
    counts = {'': counts}
    plot_stacked_histograms(counts, edges, fhand=fhand, axes=axes,
                            vlines=vlines,
                            no_interactive_win=no_interactive_win,
                            figsize=figsize, mpl_params=mpl_params,
                            bin_labels=bin_labels, **kwargs)


def plot_stacked_histograms(counts, edges, fhand=None, axes=None, vlines=None,
                            no_interactive_win=False, figsize=None,
                            mpl_params=None, bin_labels=None, **kwargs):
    'counts should be a dictionary'
    if mpl_params is None:
        mpl_params = {}

    print_figure = False
    if axes is None:
        print_figure = True
    axes, canvas, fig = _get_mplot_axes(axes, fhand, figsize=figsize)

    width = edges[1:] - edges[:-1]
    bottom = None
    for idx, (label, this_counts) in enumerate(counts.items()):
        color = cm.Paired(1. * idx / len(counts))
        axes.bar(edges[:-1], this_counts, width=width, bottom=bottom,
                 color=color, label=label, **kwargs)
        if bottom is None:
            bottom = this_counts
        else:
            bottom += this_counts

    if bin_labels is not None:
        assert len(bin_labels) == len(list(edges)) - 1
        ticks = edges[:-1] + width / 2
        axes.set_xticks(ticks)
        xticklabels = list(map(str, bin_labels))
        axes.set_xticklabels(xticklabels, rotation='vertical')

    if vlines is not None:
        ymin, ymax = axes.get_ylim()
        axes.vlines(vlines, ymin=ymin, ymax=ymax)

    for function_name, params in mpl_params.items():
        function = getattr(axes, function_name)
        function(*params.get('args', []), **params.get('kwargs', {}))

    if len(counts) > 1:
        axes.legend()

    fig.tight_layout()

    if print_figure:
        _print_figure(canvas, fhand, no_interactive_win=no_interactive_win)

    return
