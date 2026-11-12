import unittest

from src.constraints import ConstrainedDecoder


class TestNumberPrefixes(unittest.TestCase):
    def setUp(self) -> None:
        # These validators do not use the model or any instance attributes.
        self.decoder = object.__new__(ConstrainedDecoder)

    def test_integer_accepts_valid_prefixes(self) -> None:
        cases = [("", "-"), ("", "0"), ("", "265"), ("-", "34")]
        for prefix, token in cases:
            with self.subTest(prefix=prefix, token=token):
                self.assertTrue(
                    self.decoder.is_valid_integer_prefix(prefix, token)
                )

    def test_integer_rejects_invalid_prefixes(self) -> None:
        cases = [("", ""), ("", "01"), ("-", "-"), ("2", ".0"), ("", "+1")]
        for prefix, token in cases:
            with self.subTest(prefix=prefix, token=token):
                self.assertFalse(
                    self.decoder.is_valid_integer_prefix(prefix, token)
                )

    def test_number_accepts_integer_and_decimal_prefixes(self) -> None:
        cases = [
            ("", "-"), ("", "2"), ("", "2.0"), ("-", "0.5"), ("2", ".")
        ]
        for prefix, token in cases:
            with self.subTest(prefix=prefix, token=token):
                self.assertTrue(
                    self.decoder.is_valid_number_prefix(prefix, token)
                )

    def test_number_rejects_invalid_prefixes(self) -> None:
        cases = [("", ".5"), ("", "01"), ("1.2", ".3"), ("", "+2")]
        for prefix, token in cases:
            with self.subTest(prefix=prefix, token=token):
                self.assertFalse(
                    self.decoder.is_valid_number_prefix(prefix, token)
                )


class TestStringPrefixes(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = object.__new__(ConstrainedDecoder)

    def test_accepts_plain_and_escaped_text(self) -> None:
        self.assertEqual(
            self.decoder.check_str_token("NORMAL", "hello"), (True, "NORMAL")
        )
        self.assertEqual(
            self.decoder.check_str_token("NORMAL", "\\n"), (True, "NORMAL")
        )
        self.assertEqual(
            self.decoder.check_str_token("NORMAL", '\\"'), (True, "NORMAL")
        )

    def test_rejects_raw_control_character_and_unknown_escape(self) -> None:
        self.assertFalse(self.decoder.check_str_token("NORMAL", "a\nb")[0])
        self.assertFalse(self.decoder.check_str_token("NORMAL", "\\m")[0])

    def test_escape_can_span_tokens(self) -> None:
        valid, state = self.decoder.check_str_token("NORMAL", "\\")
        self.assertEqual((valid, state), (True, "ESCAPE"))
        self.assertEqual(
            self.decoder.check_str_token(state, "n"), (True, "NORMAL")
        )

    def test_unicode_escape_requires_four_hex_digits(self) -> None:
        self.assertEqual(
            self.decoder.check_str_token("NORMAL", "\\u00"), (True, "U2")
        )
        self.assertEqual(
            self.decoder.check_str_token("U2", "41"), (True, "NORMAL")
        )
        self.assertFalse(self.decoder.check_str_token("U2", "G")[0])

    def test_quote_closes_string(self) -> None:
        self.assertEqual(
            self.decoder.check_str_token("NORMAL", 'abc"'), (True, "CLOSED")
        )
        self.assertFalse(self.decoder.check_str_token("CLOSED", "x")[0])
