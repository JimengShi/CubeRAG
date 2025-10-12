def merge_return_doc_idx(returned_doc_idxs, top_k):
    """
    merge lists: [[1,2], [3,4], [5, 6]] --> [1,3,5,2,4,6]    
    """
    # Extract the first element of each inner list
    first_elements = [sublist[0] for sublist in returned_doc_idxs if len(sublist) > 0]

    # Extract all remaining elements and flatten them into a single list
    remaining_elements = [item for sublist in returned_doc_idxs if len(sublist) > 0 for item in sublist[1:]]

    # Combine the two lists
    merged_list = first_elements + remaining_elements

    # remove the duplicates but keep the original order
    seen = set()
    ordered_unique_list = []

    for item in merged_list:
        if item not in seen:
            seen.add(item)
            ordered_unique_list.append(item)

    return ordered_unique_list[:top_k]