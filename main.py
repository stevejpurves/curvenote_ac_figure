"""Generate plots for nonlocal events."""

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

MM_TO_INCHES = 1.0 / 25.4
# TWO_COLUMN = 174.0 * MM_TO_INCHES
TWO_COLUMN = 500.0 * MM_TO_INCHES

ARROW_MUTATION_SCALE = 15
AXIS_OFFSET = 10


CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "custom_PuBu", plt.get_cmap("PuBu")(np.linspace(0.3, 1, 100))
)


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
    event_dict: dict,
    subject_epoch_data: pd.DataFrame,
    linear_position_df: pd.DataFrame,
    position_df: pd.DataFrame,
    acausal_results_summary: pd.DataFrame,
    fig_path: str,
    mua=None,
    save_fig: bool = False,
    nonlocal_cmap: str = "custom",
    default_cmap: str = "PuBu",
    shading_named_color: str = "slateblue",
    arrow_color: str = "violet",
    peri_nonlocal_time: float = 0.2,
    use_manual: bool = True,
    extra_hpd: bool = False,
    show_cbar_ticks: bool = True,
    debug=False,
    *args,
    **kwargs,
):
    """Plot 1d decoding data projected to 2d with colors.

    For actual and decoded nonlocal position color nonlocal position by time.
    Illustrates sequence for a longer time snippet than just the nonlocal event.
    Also plots mua, ahead behind distance, and speed highlight the nonlocal time
    on the subplots below the position decoding plot. Keeps track of relevant
    rat, day, epoch, trial, and time information.

    Parameters
    ----------
    event_dict : dict
        Dictionary containing event information.
    subject_epoch_data : pd.DataFrame
        DataFrame containing subject epoch data.
    linear_position_df : pd.DataFrame
        DataFrame containing linear position data.
    position_df : pd.DataFrame
        DataFrame containing position data.
    acausal_results_summary : pd.DataFrame
        DataFrame containing acausal results summary data.
    fig_path : str
        Path to save the figure.
    mua : pd.DataFrame, optional
        Multiunit activity data. Default is None.
    save_fig : bool, optional
        Save the figure. Default is False.
    nonlocal_cmap : str, optional
        Colormap for nonlocal position. Default is 'custom'.
    default_cmap : str, optional
        Default colormap. Default is 'PuBu'.
    shading_named_color : str, optional
        Named color for shading. Default is 'slateblue'.
    arrow_color : str, optional
        Color for arrows. Default is 'violet'.
    peri_nonlocal_time : float, optional
        Time around nonlocal event. Default is 0.2.
    use_manual : bool, optional
        Use manual event times. Default is True.
    extra_hpd : bool, optional
    show_cbar_ticks : bool, optional
        Show colorbar ticks. Default is True.
    debug : bool, optional
        Print values during run. Default is False.
    """

    set_seaborn_opts()

    # --- Extract event information ---
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

    # --- Check if event is within valid time range ---
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

    # --- Declare figure and axes ---
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
        fontsize=14,
    )

    if debug:
        print(heading)

    # --- Plot track occupied positions ---
    lin_pos_x = linear_position_df["projected_x_position"]
    lin_pos_y = linear_position_df["projected_y_position"]

    plot_track_and_trial_path(
        axes[0], lin_pos_x, lin_pos_y, time_slice, arrow_color
    )
    axes[0].axis("off")
    add_scale_bar(axes[0])

    axes[0].set_xlim(
        lin_pos_x.min() - AXIS_OFFSET, lin_pos_x.max() + AXIS_OFFSET
    )
    axes[0].set_ylim(
        lin_pos_y.min() - AXIS_OFFSET, lin_pos_y.max() + AXIS_OFFSET
    )

    # --- Plot event times ---
    event_filtered_data = filter_by_time_range(
        subject_epoch_data_filtered_seg, event_start_t, event_stop_t
    )
    event_2d_pos = tuple(
        event_filtered_data[
            ["actual_2d_x_projected_position", "actual_2d_y_projected_position"]
        ].T.values
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
        cmap=CUSTOM_CMAP if nonlocal_cmap == "custom" else default_cmap,
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
        fig, duration, shading_named_color, CUSTOM_CMAP, show_cbar_ticks
    )

    # --- Filter by snippet (peri-event) time range ---
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

    # for ahebeh actually only want the hpd valid times. Head Pos Data??
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

    # --- Multiunit activity (MUA) and speed ---
    # mua old style which avgs across tetrodes, new one sums
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

    add_shaded_region(axes[1], shaded_times, shading_named_color)
    add_shaded_region(axes[2], shaded_times, shading_named_color)
    add_shaded_region(axes[3], shaded_times, shading_named_color)

    set_null_x_axis(axes[1], snippet_duration)
    set_null_x_axis(axes[2], snippet_duration)

    axes[3].set_xlim(0, snippet_duration)
    axes[3].set_ylim(bottom=0)

    for ax in axes.flat:
        sns.despine(offset=5, ax=ax)

    axes[1].spines["bottom"].set_visible(False)
    axes[2].spines["bottom"].set_visible(False)

    plt.subplots_adjust(hspace=0.1)

    set_axis_position(axes[1])
    set_axis_position(axes[2])
    set_axis_position(axes[3])

    # --- Save and show figure ---
    if save_fig:
        fig_name = (
            f"snippet_ex_decode_MUA_AHBEH_first{is_first_seg_of_trial}_"
            + f"{nwb_file_name[0:-5]}_{epoch}_{trial}"
            + f"_event{event_num_in_trial}_"
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


def set_axis_position(
    axis: plt.Axes, new_left: float = 0.25, new_width: float = 0.6
):
    pos = axis.get_position()
    axis.set_position([new_left, pos.y0, new_width, pos.height])


def set_null_x_axis(axis: plt.Axes, snippet_duration: float):
    """Set null axis for plotting."""
    axis.set_xlim(0, snippet_duration)
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_xlabel("")


def add_shaded_region(
    ax: plt.Axes, shaded_times: np.array, shading_named_color: str
):
    """Add a shaded region to an axis.

    Parameters
    ----------
    ax : plt.Axes
        Axis to add the shaded region to.
    shaded_times : np.array
        Array of times to shade.
    shading_named_color : str
        Named color for shading.
    """
    ax.axvspan(
        shaded_times[0],
        shaded_times[-1],
        ymin=0,
        ymax=1,
        color=shading_named_color,
        alpha=0.5,
        zorder=-100,
        linestyle="None",
    )


def add_colorbar(
    fig: plt.Figure,
    duration: float,
    shading_named_color: str,
    custom_cmap: LinearSegmentedColormap,
    show_cbar_ticks: bool,
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
    tick_length: int = 3,
    x_start: int = 200,
    y_position: int = 58,
    scale_bar_length: int = 25,
    color="lightgrey",
):
    """Add a scale bar to the plot, and label it."""
    kwargs = dict(lw=1, color=color)
    axes.plot(
        [x_start, x_start + scale_bar_length],
        [y_position, y_position],
        **kwargs,
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
        **kwargs,
    )
    axes.plot(
        [x_start + scale_bar_length, x_start + scale_bar_length],
        [y_position - tick_length, y_position + tick_length * 0],
        **kwargs,
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
        mutation_scale=ARROW_MUTATION_SCALE,
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


def filter_by_time_range(
    df: pd.DataFrame, start: float, stop: float, filter_col: str = "time"
) -> pd.DataFrame:
    """Filter a DataFrame by a time range."""
    this_col = df.index if filter_col == "index" else df[filter_col]
    return df[(this_col >= start) & (this_col <= stop)]
