#!/usr/bin/env python3

def reverse_string(s):
    return s[::-1]

if __name__ == "__main__":
    original = "debprakash"
    reversed_str = reverse_string(original)
    print(f"Original: {original}")
    print(f"Reversed: {reversed_str}")
