from __future__ import annotations

from pathlib import Path
import csv

from schema import (
    ActionLabel,
    ActionPipelineUpdate,
    ActionSegmentPrediction,
    ActionSequenceItem,
    ActionWindowPrediction,
    ActionWindowResult,
    LabeledActionSegment,
)


class ActionRecognitionPipeline:
    def __init__(
        self,
        *,
        verb_label_map: dict[int, ActionLabel],
        noun_label_map: dict[int, ActionLabel],
        score_threshold: float,
        min_duration_seconds: float,
        max_segments_per_window: int,
    ) -> None:
        self.verb_label_map = verb_label_map
        self.noun_label_map = noun_label_map
        self.score_threshold = score_threshold
        self.min_duration_seconds = min_duration_seconds
        self.max_segments_per_window = max_segments_per_window

        self.window_results: list[ActionWindowResult] = []
        self.raw_verb_segments: list[LabeledActionSegment] = []
        self.raw_noun_segments: list[LabeledActionSegment] = []
        self.merged_verb_segments: list[LabeledActionSegment] = []
        self.merged_noun_segments: list[LabeledActionSegment] = []
        self.action_sequence: list[ActionSequenceItem] = []
        self.coverage_seconds = 0.0

    @classmethod
    def from_epic_kitchens(
        cls,
        *,
        annotations_dir: str | Path,
        score_threshold: float,
        min_duration_seconds: float,
        max_segments_per_window: int,
    ) -> ActionRecognitionPipeline:
        annotations_path = Path(annotations_dir).expanduser().resolve()
        verb_label_map = cls.load_label_map(annotations_path / "EPIC_100_verb_classes.csv")
        noun_label_map = cls.load_label_map(annotations_path / "EPIC_100_noun_classes.csv")
        return cls(
            verb_label_map=verb_label_map,
            noun_label_map=noun_label_map,
            score_threshold=score_threshold,
            min_duration_seconds=min_duration_seconds,
            max_segments_per_window=max_segments_per_window,
        )

    @staticmethod
    def load_label_map(csv_path: Path) -> dict[int, ActionLabel]:
        label_map: dict[int, ActionLabel] = {}
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = ActionLabel(
                    id=int(row["id"]),
                    key=row["key"],
                    category=row.get("category", ""),
                )
                label_map[label.id] = label
        return label_map

    def process_window_predictions(
        self,
        *,
        window_index: int,
        start_seconds: float,
        end_seconds: float,
        predictions: ActionWindowPrediction,
    ) -> ActionPipelineUpdate:
        verb_segments = self._filter_segments(predictions.verb_segments, self.verb_label_map)
        noun_segments = self._filter_segments(predictions.noun_segments, self.noun_label_map)

        window_result = ActionWindowResult(
            window_index=window_index,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            verb_segments=verb_segments,
            noun_segments=noun_segments,
        )
        self.window_results.append(window_result)
        self.raw_verb_segments.extend(verb_segments)
        self.raw_noun_segments.extend(noun_segments)
        self.coverage_seconds = max(self.coverage_seconds, end_seconds)

        self.merged_verb_segments = self.merge_segments(self.raw_verb_segments)
        self.merged_noun_segments = self.merge_segments(self.raw_noun_segments)
        self.action_sequence = self.build_action_sequence(
            self.merged_verb_segments,
            self.merged_noun_segments,
            video_duration=self.coverage_seconds,
        )

        return ActionPipelineUpdate(
            window_result=window_result,
            merged_verb_segments=self.merged_verb_segments,
            merged_noun_segments=self.merged_noun_segments,
            action_sequence=self.action_sequence,
        )

    def finalize(self, video_duration: float | None = None) -> None:
        if video_duration is not None:
            self.coverage_seconds = max(self.coverage_seconds, video_duration)
        self.merged_verb_segments = self.merge_segments(self.raw_verb_segments)
        self.merged_noun_segments = self.merge_segments(self.raw_noun_segments)
        self.action_sequence = self.build_action_sequence(
            self.merged_verb_segments,
            self.merged_noun_segments,
            video_duration=self.coverage_seconds,
        )

    def _filter_segments(
        self,
        segments: list[ActionSegmentPrediction],
        label_map: dict[int, ActionLabel],
    ) -> list[LabeledActionSegment]:
        filtered: list[LabeledActionSegment] = []
        for segment in segments:
            if segment.score < self.score_threshold or segment.duration_seconds < self.min_duration_seconds:
                continue
            label = label_map.get(segment.label_id, ActionLabel(id=segment.label_id, key=f"id_{segment.label_id}"))
            filtered.append(
                LabeledActionSegment(
                    kind=segment.kind,
                    video_id=segment.video_id,
                    label_id=segment.label_id,
                    label_name=label.key,
                    label_category=label.category,
                    score=segment.score,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    duration_seconds=segment.duration_seconds,
                    window_start_seconds=segment.window_start_seconds,
                )
            )

        filtered.sort(key=lambda item: item.score, reverse=True)
        return filtered[: self.max_segments_per_window]

    def merge_segments(self, segments: list[LabeledActionSegment]) -> list[LabeledActionSegment]:
        merged: list[LabeledActionSegment] = []
        sorted_segments = sorted(
            segments,
            key=lambda item: (item.label_name, item.start_seconds, -item.score),
        )

        for segment in sorted_segments:
            matched_index: int | None = None
            for index, candidate in enumerate(merged):
                if candidate.kind != segment.kind or candidate.label_name != segment.label_name:
                    continue
                gap_seconds = segment.start_seconds - candidate.end_seconds
                if self.temporal_iou(candidate, segment) < 0.35 and gap_seconds > 0.8:
                    continue
                matched_index = index
                break

            if matched_index is None:
                merged.append(segment)
                continue

            candidate = merged[matched_index]
            start_seconds = min(candidate.start_seconds, segment.start_seconds)
            end_seconds = max(candidate.end_seconds, segment.end_seconds)
            merged[matched_index] = LabeledActionSegment(
                kind=candidate.kind,
                video_id=candidate.video_id,
                label_id=candidate.label_id,
                label_name=candidate.label_name,
                label_category=candidate.label_category,
                score=max(candidate.score, segment.score),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration_seconds=max(0.0, end_seconds - start_seconds),
                window_start_seconds=min(candidate.window_start_seconds, segment.window_start_seconds),
                merge_count=candidate.merge_count + 1,
            )

        merged.sort(key=lambda item: item.start_seconds)
        return merged

    @staticmethod
    def temporal_iou(first: LabeledActionSegment, second: LabeledActionSegment) -> float:
        left = max(first.start_seconds, second.start_seconds)
        right = min(first.end_seconds, second.end_seconds)
        intersection = max(0.0, right - left)
        if intersection <= 0:
            return 0.0

        union = first.duration_seconds + second.duration_seconds - intersection
        return 0.0 if union <= 0 else intersection / union

    @staticmethod
    def build_action_sequence(
        verb_segments: list[LabeledActionSegment],
        noun_segments: list[LabeledActionSegment],
        *,
        video_duration: float,
    ) -> list[ActionSequenceItem]:
        if video_duration <= 0:
            return []

        change_points = {0.0, video_duration}
        for segment in [*verb_segments, *noun_segments]:
            change_points.add(segment.start_seconds)
            change_points.add(segment.end_seconds)
        ordered_points = sorted(point for point in change_points if 0.0 <= point <= video_duration)

        action_sequence: list[ActionSequenceItem] = []
        for start_seconds, end_seconds in zip(ordered_points[:-1], ordered_points[1:]):
            if end_seconds - start_seconds < 0.2:
                continue

            midpoint = (start_seconds + end_seconds) / 2.0
            active_verbs = [
                segment
                for segment in verb_segments
                if segment.start_seconds <= midpoint <= segment.end_seconds
            ]
            active_nouns = [
                segment
                for segment in noun_segments
                if segment.start_seconds <= midpoint <= segment.end_seconds
            ]
            if not active_verbs and not active_nouns:
                continue

            best_verb = max(active_verbs, key=lambda item: item.score, default=None)
            best_noun = max(active_nouns, key=lambda item: item.score, default=None)
            verb_name = "none" if best_verb is None else best_verb.label_name
            noun_name = "none" if best_noun is None else best_noun.label_name
            phrase = verb_name if best_noun is None else f"{verb_name} {noun_name}"

            if action_sequence and action_sequence[-1].phrase == phrase:
                previous = action_sequence[-1]
                action_sequence[-1] = ActionSequenceItem(
                    start_seconds=previous.start_seconds,
                    end_seconds=end_seconds,
                    phrase=previous.phrase,
                    verb=previous.verb,
                    noun=previous.noun,
                )
                continue

            action_sequence.append(
                ActionSequenceItem(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    phrase=phrase,
                    verb=verb_name,
                    noun=noun_name,
                )
            )

        return action_sequence


__all__ = ["ActionRecognitionPipeline"]
