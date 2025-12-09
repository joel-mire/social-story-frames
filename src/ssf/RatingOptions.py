# Rating constants
class RatingOptions:
    IMPLAUSIBLE = "implausible or very unlikely"
    SOMEWHAT_UNLIKELY = "plausible but somewhat unlikely"
    SOMEWHAT_LIKELY = "plausible and somewhat likely"
    VERY_LIKELY = "plausible and very likely"

    @classmethod
    def all(cls):
        return [cls.IMPLAUSIBLE, cls.SOMEWHAT_UNLIKELY, cls.SOMEWHAT_LIKELY, cls.VERY_LIKELY]

    @classmethod
    def scores(cls):
        """Return 1-4 scores for each rating option (for metric computation)."""
        return {opt: i + 1 for i, opt in enumerate(cls.all())}

    @classmethod
    def ordinal_map(cls):
        """Return 0-3 ordinal mapping (for inter-annotator agreement computation)."""
        return {
            cls.IMPLAUSIBLE: 0,
            cls.SOMEWHAT_UNLIKELY: 1,
            cls.SOMEWHAT_LIKELY: 2,
            cls.VERY_LIKELY: 3,
        }

    @classmethod
    def binary_map(cls):
        """Return binary mapping: implausible/unlikely (0) vs likely/very likely (1)."""
        return {
            cls.IMPLAUSIBLE: 0,
            cls.SOMEWHAT_UNLIKELY: 0,
            cls.SOMEWHAT_LIKELY: 1,
            cls.VERY_LIKELY: 1,
        }