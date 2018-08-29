def prefix(mask):
    length = len(mask)
    length_prefix = -1
    prefix_table = [length_prefix]
    for positionInMask in range(0, length):
        while length_prefix >= 0 and mask[length_prefix] is not mask[positionInMask]:
            length_prefix = prefix_table[length_prefix]
        length_prefix = length_prefix + 1
        prefix_table.append(length_prefix)
    return prefix_table


def search_matching(mask, prefix_table, text):
    result = []
    position_in_mask = 0
    mask_length = len(mask)

    for position_in_text in range(0, len(text)):
        while position_in_mask >= 0 and text[position_in_text] != mask[position_in_mask]:
            position_in_mask = prefix_table[position_in_mask]
        position_in_mask = position_in_mask + 1

        if position_in_mask is mask_length:
            pos = position_in_text - mask_length + 1
            if pos >= 0:
                result.append(pos)
            position_in_mask = prefix_table[position_in_mask]
    return result
