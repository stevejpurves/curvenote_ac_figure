"""Generate plots for nonlocal events."""

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter1d

MM_TO_INCHES = 1.0 / 25.4
TWO_COLUMN = 174.0 * MM_TO_INCHES


def set_seaborn_opts() -> None:
    """Set seaborn options for plotting."""
    rc_params = {
        "pdf.fonttype": 42,  # Make fonts editable in Adobe Illustrator
        "ps.fonttype": 42,  # Make fonts editable in Adobe Illustrator
        "axes.labelcolor": "#222222",
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.titlesize": 9,
        "text.color": "#222222",
        "text.usetex": False,
        "figure.figsize": (6.85, 4.23),
        "xtick.major.size": 2,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.major.size": 2,
        "axes.labelpad": 0.1,
    }
    sns.set(
        style="white",
        context="paper",
        rc=rc_params,
    )


def plot_event_v3_1_mua_ahbeh_speed(
    event_dict,
    subject_epoch_data,
    linear_position_df,
    position_df,
    acausal_results_summary,
    fig_path,
    mua=None,
    save_fig=False,
    nonlocal_cmap="custom",
    default_cmap="PuBu",
    shading_named_color="slateblue",
    arrow_color="violet",
    peri_nonlocal_time=0.2,
    use_manual=True,
    extra_hpd=False,
    show_cbar_ticks=True,
    min_nonlocal_duration_s=0.02,
    between_bin_buffer_s=0.004,
    debug=False,
):
    """Plot 1d decoding data projected to 2d with colors.

    For actual and decoded nonlocal position color nonlocal position by time.
    Illustrates sequence for a longer time snippet than just the nonlocal event.
    Also plots mua, ahead behind distance, and speed highlight the nonlocal time
    on the subplots below the position decoding plot. Keeps track of relevant
    rat, day, epoch, trial, and time information.
    """

    set_seaborn_opts()

    nwb_file_name = event_dict["nwb_file_name"]
    epoch = event_dict["epoch"]
    trial = event_dict["trial"]
    is_first_seg_of_trial = event_dict["is_first_seg_of_trial"]
    event_num_in_trial = event_dict["event_num_in_trial"]

    time_slice = slice(
        event_dict["time_slice_start"], event_dict["time_slice_stop"]
    )

    event_start_t, event_stop_t = extract_event_range(event_dict, use_manual)
    duration = event_stop_t - event_start_t
    subject_epoch_data_filtered_seg = filter_epoch_data_by_event(
        event_dict, subject_epoch_data, position_df, debug=debug
    )

    if debug:
        print(f"Duration: {duration}")

    # Check masked data for out of bounds timepoints
    filtered_start, filtered_end = subject_epoch_data_filtered_seg.time.iloc[
        [0, -1]
    ]
    out_of_bounds = (
        filtered_start
        if event_start_t < filtered_start
        else filtered_end if event_stop_t > filtered_end else None
    )
    if out_of_bounds and debug:
        print(
            f"WARNING: Event start {event_start_t} is outside valid time "
            + f"{out_of_bounds}\n"
        )

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(TWO_COLUMN / 2, (TWO_COLUMN / 2) * (1.5)),
        gridspec_kw={"height_ratios": [5, 0.8, 0.8, 0.8]},
    )

    heading = (
        f"{nwb_file_name[0:-5]}, {epoch}, {trial}, E: {event_num_in_trial}, "
        + f"first seg {bool(is_first_seg_of_trial)}"
    )
    axes[0].set_title(
        heading
        + f"\nNon-local event example, duration {np.round(duration*1000,2)} ms",
        y=1.0,
        fontsize=4,
    )

    if debug:
        print(heading)

    # Plot track occupied positions
    lin_pos_x = linear_position_df["projected_x_position"]
    lin_pos_y = linear_position_df["projected_y_position"]
    plot_track_and_trial_path(
        axes[0], lin_pos_x, lin_pos_y, time_slice, arrow_color
    )
    axes[0].axis("off")
    add_scale_bar(axes[0])

    offset = 10
    axes[0].set_xlim(lin_pos_x.min() - offset, lin_pos_x.max() + offset)
    axes[0].set_ylim(lin_pos_y.min() - offset, lin_pos_y.max() + offset)

    # get event times of df
    event_filtered_data = filter_by_time_range(
        subject_epoch_data_filtered_seg, event_start_t, event_stop_t
    )
    event_2d_pos = tuple(
        event_filtered_data[
            ["actual_2d_x_projected_position", "actual_2d_y_projected_position"]
        ].T.values
    )

    # create custom colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_PuBu", plt.get_cmap("PuBu")(np.linspace(0.3, 1, 100))
    )

    # plot nonlocal actual and mental pos
    axes[0].scatter(
        *event_2d_pos,
        s=35,
        marker="o",
        c="magenta",  # facecolors="none",
        label="Rat pos.",
    )
    axes[0].scatter(
        *event_2d_pos,
        s=50,
        marker="o",
        c=event_filtered_data.time - event_start_t,
        cmap=custom_cmap if nonlocal_cmap == "custom" else default_cmap,
        label="Represented\nnon-local pos.",
        vmin=0,
    )
    # legend for actual and mental pos
    legend = axes[0].legend(
        bbox_to_anchor=(-0.03, 0.5), loc="lower left", frameon=False
    )
    text_colors = ["dimgrey", "magenta", shading_named_color]
    for label, color in zip(legend.get_texts(), text_colors):
        label.set_color(color)

    add_colorbar(
        fig, duration, shading_named_color, custom_cmap, show_cbar_ticks
    )

    # filter by snippet time range
    snippet_time_start = event_start_t - peri_nonlocal_time
    snippet_time_stop = event_stop_t + peri_nonlocal_time
    snippet_duration = snippet_time_stop - snippet_time_start
    snippet_times = (
        acausal_results_summary.loc[snippet_time_start:snippet_time_stop].index
        - snippet_time_start
    )
    snip_kwargs = dict(  # for filter_by_time_range function
        start=snippet_time_start, stop=snippet_time_stop, filter_col="index"
    )

    if debug:
        print(len(snippet_times))

    # for ahebeh actually only want the hpd valid times
    if extra_hpd:
        acausal_results_summary_hpd = acausal_results_summary[
            acausal_results_summary.spatial_coverage_50_hpd <= 50
        ]
        snippet_times_hpd = (
            acausal_results_summary_hpd.loc[
                snippet_time_start:snippet_time_stop
            ].index
            - snippet_time_start
        )
        ahbeh_snippet_hpd = filter_by_time_range(
            df=acausal_results_summary_hpd.ahead_behind_distance, **snip_kwargs
        )
        axes[2].scatter(
            snippet_times_hpd,
            ahbeh_snippet_hpd,
            color="darkgrey",
            alpha=1,
            s=4,
            zorder=5000,
        )
    else:
        ahbeh_snippet = filter_by_time_range(
            df=acausal_results_summary.ahead_behind_distance, **snip_kwargs
        )
        axes[2].scatter(
            snippet_times, ahbeh_snippet, color="darkgrey", alpha=1, s=4
        )

    axes[2].set_ylabel("cm")
    axes[2].axhline(
        0, color="magenta", alpha=0.3, linewidth=1, zorder=0, linestyle="-"
    )

    # mua old style which avgs across tets, new one sums
    if mua is not None:
        mua_snippet = filter_by_time_range(df=mua.firing_rate, **snip_kwargs)
        axes[1].fill_between(
            snippet_times, mua_snippet, color="dimgrey", alpha=1
        )
    else:
        mua_tet_snippet = filter_by_time_range(
            df=acausal_results_summary.multiunit_firing_rate, **snip_kwargs
        )
        axes[1].fill_between(
            snippet_times, mua_tet_snippet, color="darkgrey", alpha=1
        )

    axes[1].set_ylabel("MUA\n(spikes/s)")
    _, ymax = axes[1].get_ylim()
    ymax1 = int(np.ceil(ymax / 100.0)) * 100
    axes[1].set_ylim([0, ymax1])
    axes[1].set_yticks([0, ymax1])

    head_speed_snippet = filter_by_time_range(
        df=acausal_results_summary.head_speed, **snip_kwargs
    )
    axes[3].fill_between(
        snippet_times, head_speed_snippet, color="darkgrey", alpha=1
    )
    axes[3].set_ylabel("Speed\n(cm/s)")
    axes[3].set_xlabel("Time (s)", labelpad=6)

    shaded_times = (
        acausal_results_summary.loc[event_start_t:event_stop_t].index
        - snippet_time_start
    )

    if axes[3].get_ylim()[1] <= 55:
        axes[3].set_ylim(0, 55)

    axes[1].axvspan(
        shaded_times[0],
        shaded_times[-1],
        ymin=0,
        ymax=1,
        color=shading_named_color,  # edgecolor="none",
        alpha=0.5,
        zorder=-100,
        linestyle="None",
    )
    axes[3].axvspan(
        shaded_times[0],
        shaded_times[-1],
        ymin=0,
        ymax=1,
        color=shading_named_color,  # edgecolor="none",
        alpha=0.5,
        zorder=-100,
        linestyle="None",
    )
    axes[2].axvspan(
        shaded_times[0],
        shaded_times[-1],
        ymin=0,
        ymax=1,
        color=shading_named_color,  # edgecolor="none",
        alpha=0.5,
        zorder=-100,
        linestyle="None",
    )

    axes[1].set_xlim(0, snippet_duration)
    axes[1].set_xticks([])
    axes[1].set_xticklabels([])
    axes[1].set_xlabel("")

    axes[2].set_xlim(0, snippet_duration)
    axes[2].set_xticks([])
    axes[2].set_xticklabels([])
    axes[2].set_xlabel("")

    axes[3].set_xlim(0, snippet_duration)
    axes[3].set_ylim(bottom=0)

    for ax in axes.flat:
        sns.despine(offset=5, ax=ax)

    axes[1].spines["bottom"].set_visible(False)
    axes[2].spines["bottom"].set_visible(False)

    plt.subplots_adjust(hspace=0.1)

    pos_ax1 = axes[1].get_position()
    pos_ax2 = axes[2].get_position()
    pos_ax3 = axes[3].get_position()
    new_left = 0.25
    new_width = 0.6

    axes[1].set_position([new_left, pos_ax1.y0, new_width, pos_ax1.height])
    axes[2].set_position([new_left, pos_ax2.y0, new_width, pos_ax2.height])
    axes[3].set_position([new_left, pos_ax3.y0, new_width, pos_ax3.height])

    if save_fig:
        fig_name = (
            f"snippet_ex_decode_MUA_AHBEH_first{is_first_seg_of_trial}_"
            + f"{nwb_file_name[0:-5]}_{epoch}_{trial}"
            + f"_event{event_num_in_trial}_"
            + f"mindur{min_nonlocal_duration_s}_"
            + f"binbuff{between_bin_buffer_s}_"
            + f"perievent{peri_nonlocal_time}_"
            + f"shade{shading_named_color}_{event_start_t}_{event_stop_t}"
        )

        plt.savefig(
            f"{fig_path}{fig_name}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.5,
            dpi=300,
        )

    plt.show()


def add_colorbar(
    fig, duration, shading_named_color, custom_cmap, show_cbar_ticks
):
    """Add a colorbar to the figure."""
    cbar_ax2 = fig.add_axes([0.45, 0.48, 0.15, 0.02])
    norm = mcolors.Normalize(vmin=0, vmax=duration)
    scatter2bar = cm.ScalarMappable(norm=norm, cmap=custom_cmap)
    cbar2 = plt.colorbar(scatter2bar, cax=cbar_ax2, orientation="horizontal")

    cbar2.set_label("ms", color=shading_named_color, alpha=1)
    cbar2.outline.set_visible(False)
    cbar2.ax.spines["left"].set_visible(False)
    cbar2.ax.tick_params(color=shading_named_color)

    if not show_cbar_ticks:
        cbar2.ax.xaxis.set_ticks_position("none")

    cbar2.set_ticks([0, duration])
    cbar2.set_ticklabels([0, int(np.round(1000 * duration, 0))])

    for label in cbar2.ax.get_xticklabels():
        label.set_color(shading_named_color)


def add_scale_bar(
    axes,
    tick_length=3,
    x_start=200,
    y_position=58,
    scale_bar_length=25,
    color="lightgrey",
):
    """Add a scale bar to the plot, and label it."""
    axes.plot(
        [x_start, x_start + scale_bar_length],
        [y_position, y_position],
        color=color,
        lw=1,
    )
    axes.text(
        x_start + scale_bar_length / 2,
        y_position - 6,
        f"{scale_bar_length} cm",
        ha="center",
        va="top",
        color=color,
    )
    axes.plot(
        [x_start, x_start],
        [y_position - tick_length, y_position + tick_length * 0],
        color=color,
        lw=1,
    )
    axes.plot(
        [x_start + scale_bar_length, x_start + scale_bar_length],
        [y_position - tick_length, y_position + tick_length * 0],
        color=color,
        lw=1,
    )


def plot_track_and_trial_path(
    axes, lin_pos_x, lin_pos_y, time_slice, arrow_color
):
    """Plot the track and trial path on the main axes."""
    axes.scatter(
        lin_pos_x.iloc[::50],
        lin_pos_y.iloc[::50],
        s=50,
        color="lightgrey",
        zorder=0,
        label="_",
    )
    axes.plot(
        lin_pos_x.iloc[time_slice],
        lin_pos_y.iloc[time_slice],
        color=arrow_color,
        lw=1,
        zorder=1,
        label="Trial path",
        linestyle="--",
    )

    # Draw arrow to indicate motion direction over last 50 points
    arrow_start = (
        lin_pos_x.iloc[time_slice].values[-50],
        lin_pos_y.iloc[time_slice].values[-50],
    )
    arrow_end = (
        lin_pos_x.iloc[time_slice].values[-1],
        lin_pos_y.iloc[time_slice].values[-1],
    )
    arrow = FancyArrowPatch(
        arrow_start,
        arrow_end,
        arrowstyle="-|>",
        color=arrow_color,
        lw=1,
        mutation_scale=15,
        zorder=20000,
    )
    axes.add_patch(arrow)
    axes.set_aspect("equal")


def extract_event_range(event_dict, use_manual=True):
    """Get event start and stop times from event_dict."""
    e_start_col, e_stop_col = "event_start_t", "event_stop_t"
    if use_manual:  # if use_manual, cols are 'event_{X}_t_manual'
        e_start_col += "_manual"
        e_stop_col += "_manual"
    return event_dict[e_start_col], event_dict[e_stop_col]


def filter_epoch_data_by_event(
    event_dict, subject_epoch_data, position_df, debug=False
):
    """Filter subject epoch data based on event and position time ranges."""
    time_slice = slice(
        event_dict["time_slice_start"], event_dict["time_slice_stop"]
    )
    start_t, stop_t = position_df.iloc[time_slice].index[[0, -1]]

    subject_epoch_data = subject_epoch_data.reset_index()
    subject_epoch_data_filtered = filter_by_time_range(
        df=subject_epoch_data, start=start_t, stop=stop_t
    )

    is_first_seg_of_trial = event_dict["is_first_seg_of_trial"]
    is_last_seg_of_trial = event_dict["is_last_seg_of_trial"]

    if not is_first_seg_of_trial ^ is_last_seg_of_trial:
        raise ValueError(
            "only first OR last seg can be true, and one must be true"
        )

    trial_mask = (
        subject_epoch_data_filtered.is_first_seg_of_trial
        if is_first_seg_of_trial
        else subject_epoch_data_filtered.is_last_seg_of_trial
    )
    subject_epoch_data_filtered_seg = subject_epoch_data_filtered[trial_mask]

    if debug:
        filtered_start, filtered_end = (
            subject_epoch_data_filtered_seg.time.iloc[[0, -1]]
        )
        print(f"Filtered time range: {filtered_start} to {filtered_end}")

    return subject_epoch_data_filtered_seg


def get_multiunit_population_firing_rate_sum(
    multiunit: np.ndarray,
    sampling_frequency: float,
    smoothing_sigma: float = 0.015,
):
    """Calculates the multiunit population firing rate.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_signals)
        Binary array of multiunit spike times.
    sampling_frequency : float
        Number of samples per second.
    smoothing_sigma : float or np.timedelta
        Amount to smooth the firing rate over time. The default is
        given assuming time is in units of seconds.

    Returns
    -------
    multiunit_population_firing_rate : ndarray, shape (n_time,)
    """
    return gaussian_smooth(
        multiunit.sum(axis=1) * sampling_frequency,
        smoothing_sigma,
        sampling_frequency,
    )


def filter_by_time_range(
    df: pd.DataFrame, start: float, stop: float, filter_col: str = "time"
) -> pd.DataFrame:
    """Filter a DataFrame by a time range."""
    this_col = df.index if filter_col == "index" else df[filter_col]
    return df[(this_col >= start) & (this_col <= stop)]


def gaussian_smooth(data, sigma, sampling_frequency, axis=0, truncate=8):
    """1D convolution of the data with a Gaussian.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    """
    return gaussian_filter1d(
        data,
        sigma * sampling_frequency,
        truncate=truncate,
        axis=axis,
        mode="constant",
    )
