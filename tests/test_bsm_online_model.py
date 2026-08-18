import pytest

pytest.importorskip("experiments", reason="experiments module not available")

from experiments.blind_spectrum_monitoring.online_spectrum_model import (
    BandwidthClassHypothesis,
    OnlineSpectrumModel,
    OnlinePeakTracker,
    TrackStateDecoder,
    TrackState,
    evaluate_model_on_scans,
)
from src.tasks.blind_spectrum_monitoring.dgp import (
    ChannelDef,
    DetectedPeak,
    ScanInstance,
    SpectrumDGP,
)


def _scan(scan_idx: int, peaks: list[DetectedPeak]) -> ScanInstance:
    return ScanInstance(
        scan_idx=scan_idx,
        detected_peaks=peaks,
        ground_truth=[],
        interference_freqs=[],
    )


class TestOnlineSpectrumModel:
    def test_tracker_and_decoder_are_separate_layers(self):
        tracker = OnlinePeakTracker(band_width=120.0)
        decoder = TrackStateDecoder(band_width=120.0)
        scans = [
            _scan(0, [DetectedPeak(29.2, -38.0, 9.5, "channel")]),
            _scan(1, [DetectedPeak(28.8, -39.5, 10.2, "channel")]),
        ]

        for scan in scans:
            tracker_snapshot = tracker.update(scan)

        assert tracker_snapshot.confirmed_tracks == 1
        latent = decoder.get_latent_channels(
            tracker.tracks,
            scan_index=tracker.scan_index,
        )
        assert len(latent) == 1
        assert abs(latent[0].center_freq_mhz - 29.0) < 1.0
        assert abs(latent[0].bandwidth_mhz - 9.85) < 1.0

    def test_repeated_observations_confirm_single_track(self):
        model = OnlineSpectrumModel(band_width=120.0)
        scans = [
            _scan(0, [DetectedPeak(29.2, -38.0, 9.5, "channel")]),
            _scan(1, [DetectedPeak(28.6, -40.0, 10.4, "channel")]),
            _scan(2, [DetectedPeak(29.1, -39.2, 10.1, "channel")]),
        ]

        for scan in scans:
            snapshot = model.update(scan)

        assert snapshot.confirmed_tracks == 1
        assert len(snapshot.latent_channels) == 1
        channel = snapshot.latent_channels[0]
        assert abs(channel.center_freq_mhz - 29.0) < 1.0
        assert abs(channel.bandwidth_mhz - 10.0) < 1.5
        assert channel.hits == 3

    def test_single_false_birth_is_pruned_if_not_repeated(self):
        model = OnlineSpectrumModel(band_width=120.0)
        model.update(_scan(0, [DetectedPeak(63.0, -46.0, 4.1, "noise")]))

        for idx in range(1, 7):
            snapshot = model.update(_scan(idx, []))

        assert snapshot.total_tracks == 0
        assert snapshot.latent_channels == []

    def test_two_channels_keep_distinct_latent_ids(self):
        model = OnlineSpectrumModel(band_width=120.0)
        scans = [
            _scan(
                0,
                [
                    DetectedPeak(18.8, -39.0, 5.2, "channel"),
                    DetectedPeak(53.4, -37.5, 10.2, "channel"),
                ],
            ),
            _scan(
                1,
                [
                    DetectedPeak(17.4, -40.1, 4.8, "channel"),
                    DetectedPeak(52.1, -38.4, 9.9, "channel"),
                ],
            ),
            _scan(
                2,
                [
                    DetectedPeak(17.9, -39.6, 5.0, "channel"),
                    DetectedPeak(53.0, -39.0, 10.5, "channel"),
                ],
            ),
        ]

        for scan in scans:
            snapshot = model.update(scan)

        assert snapshot.confirmed_tracks == 2
        ids = [channel.id for channel in snapshot.latent_channels]
        centers = [channel.center_freq_mhz for channel in snapshot.latent_channels]
        assert len(set(ids)) == 2
        assert centers[0] < centers[1]
        assert abs(centers[0] - 18.0) < 1.5
        assert abs(centers[1] - 53.0) < 1.5

    def test_dgp_evaluation_recovers_simple_single_channel(self):
        channel_defs = [ChannelDef(id=1, slot=0, power_dbm=-40.0)]
        dgp = SpectrumDGP(
            W=10.0,
            G=14.0,
            channels=channel_defs,
            n_instances=6,
            seed=7,
            p_active=1.0,
            p_miss=0.0,
            p_interference=0.0,
            p_false_alarm=0.0,
            freq_noise=1.0,
            power_noise=1.0,
            center_jitter=0.0,
            band_width=120.0,
        )
        scans = dgp.generate()
        model = OnlineSpectrumModel(band_width=120.0, measurement_freq_sigma=1.5)

        results = evaluate_model_on_scans(
            scans,
            model=model,
            all_latent_channel_defs=channel_defs,
            W=10.0,
            G=14.0,
        )

        assert results[-1]["score"] > 0.8
        assert results[-1]["n_confirmed"] == 1

    def test_decoder_stays_single_class_without_clear_split(self):
        decoder = TrackStateDecoder(band_width=120.0)
        tracks = [
            TrackState(
                id=1,
                center_mean=5.0,
                center_var=1.0,
                width_mean=9.7,
                width_var=0.5,
                power_mean=-40.0,
                hits=5,
                misses=0,
                first_seen_scan=1,
                last_seen_scan=5,
                recent_widths=[9.8, 9.5, 10.0, 9.6],
            ),
            TrackState(
                id=2,
                center_mean=29.0,
                center_var=1.0,
                width_mean=10.3,
                width_var=0.5,
                power_mean=-40.0,
                hits=5,
                misses=0,
                first_seen_scan=1,
                last_seen_scan=5,
                recent_widths=[10.1, 10.4, 10.2, 10.3],
            ),
            TrackState(
                id=3,
                center_mean=53.0,
                center_var=1.0,
                width_mean=9.9,
                width_var=0.5,
                power_mean=-40.0,
                hits=5,
                misses=0,
                first_seen_scan=1,
                last_seen_scan=5,
                recent_widths=[9.8, 10.0, 9.9, 10.1],
            ),
        ]

        hypotheses = decoder.infer_bandwidth_classes(tracks, scan_index=5)
        assert len(hypotheses) == 1
        assert abs(hypotheses[0].center_mhz - 10.0) < 0.5

    def test_decoder_introduces_second_class_when_supported(self):
        decoder = TrackStateDecoder(band_width=120.0)
        tracks = [
            TrackState(
                id=1,
                center_mean=5.0,
                center_var=1.0,
                width_mean=10.1,
                width_var=0.5,
                power_mean=-40.0,
                hits=6,
                misses=0,
                first_seen_scan=1,
                last_seen_scan=6,
                recent_widths=[9.8, 10.0, 10.3, 10.2],
            ),
            TrackState(
                id=2,
                center_mean=29.0,
                center_var=1.0,
                width_mean=9.8,
                width_var=0.5,
                power_mean=-40.0,
                hits=6,
                misses=0,
                first_seen_scan=1,
                last_seen_scan=6,
                recent_widths=[9.6, 9.9, 10.0, 9.8],
            ),
            TrackState(
                id=3,
                center_mean=17.0,
                center_var=1.0,
                width_mean=5.2,
                width_var=0.5,
                power_mean=-40.0,
                hits=5,
                misses=0,
                first_seen_scan=2,
                last_seen_scan=6,
                recent_widths=[4.9, 5.1, 5.3, 5.2],
            ),
            TrackState(
                id=4,
                center_mean=65.0,
                center_var=1.0,
                width_mean=4.8,
                width_var=0.5,
                power_mean=-40.0,
                hits=5,
                misses=0,
                first_seen_scan=2,
                last_seen_scan=6,
                recent_widths=[4.7, 4.9, 5.0, 4.8],
            ),
        ]

        hypotheses = decoder.infer_bandwidth_classes(tracks, scan_index=6)
        assert len(hypotheses) == 2
        centers = sorted(h.center_mhz for h in hypotheses)
        assert abs(centers[0] - 5.0) < 0.5
        assert abs(centers[1] - 10.0) < 0.5

    def test_shrink_bandwidth_pulls_low_support_track_toward_class_center(self):
        decoder = TrackStateDecoder(band_width=120.0)
        track = TrackState(
            id=7,
            center_mean=77.0,
            center_var=1.0,
            width_mean=8.4,
            width_var=0.8,
            power_mean=-40.0,
            hits=2,
            misses=0,
            first_seen_scan=4,
            last_seen_scan=5,
            recent_widths=[8.2, 8.6],
        )
        hypothesis = BandwidthClassHypothesis(
            center_mhz=10.0,
            member_track_ids=[7],
            confidence=0.9,
        )

        shrunk = decoder.shrink_bandwidth(
            track,
            scan_index=5,
            bandwidth_classes=[hypothesis],
        )

        assert shrunk > track.robust_width_estimate()
        assert shrunk < 10.0

    def test_decoder_hypothesizes_missing_internal_channel_from_stable_pattern(self):
        decoder = TrackStateDecoder(band_width=120.0)
        tracks = [
            TrackState(
                id=1,
                center_mean=5.0,
                center_var=1.0,
                width_mean=10.0,
                width_var=0.5,
                power_mean=-40.0,
                hits=6,
                misses=0,
                first_seen_scan=1,
                last_seen_scan=6,
                recent_widths=[9.8, 10.0, 10.1, 10.0],
            ),
            TrackState(
                id=2,
                center_mean=53.0,
                center_var=1.0,
                width_mean=10.0,
                width_var=0.5,
                power_mean=-40.0,
                hits=6,
                misses=0,
                first_seen_scan=1,
                last_seen_scan=6,
                recent_widths=[9.9, 10.0, 10.2, 10.1],
            ),
            TrackState(
                id=3,
                center_mean=77.0,
                center_var=1.0,
                width_mean=9.8,
                width_var=0.5,
                power_mean=-40.0,
                hits=6,
                misses=0,
                first_seen_scan=1,
                last_seen_scan=6,
                recent_widths=[9.6, 9.8, 10.0, 9.9],
            ),
            TrackState(
                id=4,
                center_mean=101.0,
                center_var=1.0,
                width_mean=10.1,
                width_var=0.5,
                power_mean=-40.0,
                hits=6,
                misses=0,
                first_seen_scan=1,
                last_seen_scan=6,
                recent_widths=[10.0, 10.2, 10.1, 10.2],
            ),
        ]

        latent = decoder.get_latent_channels(tracks, scan_index=6)
        hypothesized = [
            channel for channel in latent if channel.source == "hypothesized_grid"
        ]

        assert len(hypothesized) == 1
        assert abs(hypothesized[0].center_freq_mhz - 29.0) < 1.0
        assert abs(hypothesized[0].bandwidth_mhz - 10.0) < 0.5

    def test_decoder_does_not_hypothesize_unseen_channels_from_two_tracks(self):
        decoder = TrackStateDecoder(band_width=120.0)
        tracks = [
            TrackState(
                id=1,
                center_mean=17.0,
                center_var=1.0,
                width_mean=5.0,
                width_var=0.5,
                power_mean=-40.0,
                hits=5,
                misses=0,
                first_seen_scan=1,
                last_seen_scan=5,
                recent_widths=[4.8, 5.0, 5.2, 5.0],
            ),
            TrackState(
                id=2,
                center_mean=65.0,
                center_var=1.0,
                width_mean=5.1,
                width_var=0.5,
                power_mean=-40.0,
                hits=5,
                misses=0,
                first_seen_scan=1,
                last_seen_scan=5,
                recent_widths=[4.9, 5.1, 5.2, 5.0],
            ),
        ]

        latent = decoder.get_latent_channels(tracks, scan_index=5)
        hypothesized = [
            channel for channel in latent if channel.source == "hypothesized_grid"
        ]

        assert hypothesized == []
