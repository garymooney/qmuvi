from qmuvi.data_manager import extract_natural_number_from_string_end


def test_extract_natural_number_from_string_end():
    # Test case with a number at the end of the string
    s = "hello123"
    expected = 123
    assert extract_natural_number_from_string_end(s) == expected
    s = "35abc987"
    expected = 987
    assert extract_natural_number_from_string_end(s) == expected
    s = "weg35abc666"
    expected = 666
    assert extract_natural_number_from_string_end(s) == expected
    s = "weg35abc6.45"
    expected = 45
    assert extract_natural_number_from_string_end(s) == expected
    s = "weg35a_bc6.4_5"
    expected = 5
    assert extract_natural_number_from_string_end(s) == expected

    # Test case with no number at the end of the string
    s = "hello"
    expected = None
    assert extract_natural_number_from_string_end(s) == expected

    # Test case with empty string
    s = ""
    expected = None
    assert extract_natural_number_from_string_end(s) == expected

    # Test case with None as input
    s = None
    expected = None
    assert extract_natural_number_from_string_end(s) == expected

    # Test case with no number at end and zero_if_none = True
    s = "hello"
    expected = 0
    assert extract_natural_number_from_string_end(s, True) == expected

    # Test case with a windows path
    s = "C:\\Users\\user\\Documents\\test\\test-1"
    expected = 1
    assert extract_natural_number_from_string_end(s) == expected

    # Test case with a linux path
    s = "/usr/bin/bc6.4_5"
    expected = 5
    assert extract_natural_number_from_string_end(s) == expected
