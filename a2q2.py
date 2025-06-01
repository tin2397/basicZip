import sys
from bitarray import bitarray
import heapq

class Node:
    """
    A Node class representing a character and its frequency in the Huffman tree.

    Attributes:
        char (str): The character represented by the node (None for internal nodes).
        freq (int): The frequency of the character in the text.
        left (Node): The left child node in the Huffman tree (None if not applicable).
        right (Node): The right child node in the Huffman tree (None if not applicable).
        direction (str): The direction taken from parent node to reach this node (empty string by default).
    """
    def __init__(self, char, freq, left=None, right=None):
        """
        Initializes a Node object with a character, its frequency, and optional left and right children.
        """
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
        self.direction = ''

    def __lt__(self, other):
        """
        Defines the less-than operator for comparing two nodes based on frequency.
        This is used to sort nodes in a priority queue.
        """
        return self.freq < other.freq

# Huffman Tree class
class Huffman_Tree:
    """
    A Huffman Tree class used to build a Huffman encoding tree from character frequencies
    and generate binary codes for each character.

    Attributes:
        freq_dict (dict): A dictionary mapping characters to their frequencies.
        heap (list): A priority queue (min-heap) used to build the Huffman tree.
        codes (dict): A dictionary mapping characters to their Huffman binary codes.
        root (Node): The root node of the Huffman tree.
    """
    def __init__(self, freq_dict):
        """
        Initializes a Huffman_Tree object with a frequency dictionary.
        
        Args:
            freq_dict (dict): A dictionary mapping characters to their frequencies.
        """
        self.freq_dict = freq_dict
        self.heap = []
        self.codes = {}
        self.build_heap()
        self.build_huffman_tree()
        self.generate_codes()

    def build_heap(self):
        """
        Builds the priority queue (min-heap) from the frequency dictionary.
        Each entry in the heap is a Node representing a character and its frequency.
        """
        # Build the heap from the frequency dictionary
        for char, freq in self.freq_dict.items():
            node = Node(char, freq)
            heapq.heappush(self.heap, node)

    def build_huffman_tree(self):
        """
        Builds the Huffman tree by repeatedly combining the two nodes with the lowest frequencies.
        The final tree's root is stored in `self.root`.
        """
        # Build the Huffman Tree by combining nodes in the heap
        while len(self.heap) > 1:
            # Remove two nodes with the lowest frequencies
            left = heapq.heappop(self.heap)
            right = heapq.heappop(self.heap)

            # Create a new internal node with the combined frequency
            merged = Node(None, left.freq + right.freq, left, right)

            # Push the new node back into the heap
            heapq.heappush(self.heap, merged)

        # The root of the tree is the remaining node in the heap
        self.root = heapq.heappop(self.heap)

    def generate_codes_helper(self, node, current_code):
        """
        A recursive helper function that generates binary Huffman codes for each character.

        Args:
            node (Node): The current node in the Huffman tree.
            current_code (str): The binary code generated so far for the current traversal.
        """

        if node is None:
            return
        
        # If it's a leaf node (i.e., node.char is not None), store the character and its code
        if node.char is not None:
            self.codes[node.char] = current_code
            return
        
        # Recurse for the left and right children, appending '0' for left and '1' for right
        self.generate_codes_helper(node.left, current_code + "0")
        self.generate_codes_helper(node.right, current_code + "1")

    def generate_codes(self):
        """
        Generates Huffman codes for all characters by calling the recursive helper function
        starting from the root of the Huffman tree.
        """
        self.generate_codes_helper(self.root, "")

    def get_huffman_codes(self, char):
        """
        ***Mini-helper function to get the Huffman code for a specific character.***

        Retrieves the Huffman code for a specific character.
        
        Args:
            char (str): The character whose Huffman code is required.
        
        Returns:
            str: The binary Huffman code for the specified character.
        """
        return self.codes[char]


def huffman_encode(huffman_tree, freq_dict):
    """
    Encodes the Huffman tree and character frequencies into a compressed bit string using Elias omega coding.

    Args:
        huffman_tree (Huffman_Tree): The Huffman tree object that contains the Huffman codes for each character.
        freq_dict (dict): A dictionary mapping characters to their frequencies in the original input.

    Returns:
        str: A bit string representing the encoded Huffman tree and codes, ready for compression.
    
    The encoding process includes:
    1. Encoding the number of distinct characters using Elias omega encoding.
    2. For each character:
        - Encoding the character in 8-bit ASCII.
        - Encoding the length of its Huffman code using Elias omega encoding.
        - Adding the actual Huffman code as a bitarray.
    """

    # Encode the number of distinct characters
    encoded_distinct_chars = elias_omega_non_negative_encode(len(freq_dict))

    encoded = bitarray()

    # Add the encoded number of distinct characters to the encoded string
    encoded.extend(encoded_distinct_chars)
    # Encode the Huffman codes for each character
    for char, code in huffman_tree.codes.items():
        # Encode the distinct character with 8 bits ASCII
        encoded.extend(char_encode(char))
        # Encode the length of the Huffman code
        encoded.extend(elias_omega_non_negative_encode(len(code)))

        # Add the Huffman code
        encoded.extend(bitarray(code))

    return encoded.to01()


def character_frequency(string):
    """
    Computes the frequency of each character in the given string.

    Args:
        string (str): The input string from which to count character frequencies.

    Returns:
        dict: A dictionary where the keys are characters from the input string, and the values are their corresponding frequencies.
    
    The function iterates over each character in the input string. If the character is already in the dictionary, its count is incremented. 
    Otherwise, a new entry is added to the dictionary with an initial count of 1.
    """
    freq_dict = {}
    for char in string:
        if char in freq_dict:
            freq_dict[char] += 1
        else:
            freq_dict[char] = 1
    return freq_dict


def elias_omega_non_negative_encode(n):
    """
    Encodes a non-negative integer using Elias Omega encoding.

    Args:
        n (int): The non-negative integer to encode.

    Returns:
        str: A binary string representing the Elias Omega encoding of the input integer.

    Raises:
        ValueError: If the input `n` is negative.

    The encoding process works by:
    1. Adjusting the input `n` by adding 1 to ensure non-negative values are handled correctly.
    2. Converting the adjusted integer to a binary string and appending it to the encoded result.
    3. Recursively prepending the length of the binary string minus 1, in binary format, until a single bit remains.
    """
    # Check if the input is a non-negative integer
    if n < 0:
        raise ValueError("[from Elias Omega]: The input must be a non-negative integer.")

    # Increment the input by 1 to make it non-negative
    non_negative_n = n + 1
    encoded = bitarray()
    # Convert the input to binary
    binary_n = bin(non_negative_n)[2:]
    # Encode the binary string using Elias Omega encoding
    encoded.extend(binary_n)
    
    while len(binary_n) > 1:
        binary_n = bin(len(binary_n) - 1)[2:]
        
        binary_n = '0' + binary_n[1:]
        new_binary_n = bitarray(binary_n)
        encoded = new_binary_n + encoded

    return encoded.to01()


def lz77_compress(data):
    """
    Compress data using the LZ77 algorithm with an infinitely long search window and lookahead buffer.

    Args:
        data (str): The input string to be compressed.

    Returns:
        list: A list of tuples where each tuple represents a token in the compressed format.
              Each tuple is in the form (offset, match_length, next_char), where:
              - `offset` is the distance from the current position to the start of the match in the search window.
              - `match_length` is the length of the longest match found.
              - `next_char` is the character following the matched sequence (if no match, it's the current literal character).

    The compression process works by:
    1. Searching for the longest match of the current position in the lookback window (from the start of the data).
    2. If no match is found, the literal character at the current position is encoded.
    3. If a match is found, the `offset` (distance to the start of the match), `match_length` (length of the match), and the next character after the match are encoded.
    4. If the match extends to the end of the input, the next character is set as the last character of the data, 
        and the match length is adjusted accordingly (minus 1 and add the last character as next_char).
    """
    compressed_data = []
    i = 0  # Current position in the input data

    while i < len(data):
        # Look for the longest match in the search window (which stretches to the start of the data)
        match_length = 0
        offset = 0
        for j in range(0, i):  # Search window goes from the start of the text to position i
            length = 0
            while (i + length < len(data)) and (data[j + length] == data[i + length]):
                length += 1
            if length > match_length:
                offset = i - j
                match_length = length
                
        if match_length == 0:
            # No match, just add the literal character
            compressed_data.append((0, 0, data[i]))
            i += 1  # Move to the next character
        else:
            # If we found a match, record the offset, length, and the next character
            if i + match_length < len(data):
                next_char = data[i + match_length]
                compressed_data.append((offset, match_length, next_char))
            else:
            # If the match extends to the end of the data, we don't have a next character, 
            # therefore we decrement the match length by 1 and set the next character to the last character in the data
                next_char = data[-1]
                new_match_length = match_length - 1
                compressed_data.append((offset, new_match_length, next_char))
            
            i += match_length + 1  # Move past the matched sequence
    return compressed_data


def lz77_encode(compressed_data, huffman_tree):
    """
    Encodes the compressed LZ77 data into a binary string using Elias Omega encoding for offsets and lengths,
    and Huffman encoding for the next character.

    Args:
        compressed_data (list): A list of tuples, where each tuple is in the form (offset, length, next_char),
                                representing the compressed data from the LZ77 algorithm.
        huffman_tree (Huffman_Tree): The Huffman tree object that provides Huffman codes for characters.

    Returns:
        str: A binary string representing the encoded compressed data.

    The encoding process works by:
    1. Encoding the `offset` and `length` of each LZ77 token using Elias Omega encoding.
    2. Encoding the `next_char` (literal or the character following the match) using Huffman encoding.
    3. Concatenating these encodings into a single bitarray, which is then converted to a binary string for output.
    """
    encoded = bitarray()
    for offset, length, next_char in compressed_data:
        # Encode the offset and length using Elias Omega encoding
        encoded.extend(elias_omega_non_negative_encode(offset))
        encoded.extend(elias_omega_non_negative_encode(length))
        # Encode the next character
        encoded.extend(huffman_tree.get_huffman_codes(next_char))
    return encoded.to01()


def char_encode(file_name):
    """
    Encodes each character of the input string into an 8-bit binary representation (ASCII format).

    Args:
        file_name (str): The input string (file name or any other string) to be encoded.

    Returns:
        str: A binary string representing the 8-bit binary encoding of each character in the input string.

    The encoding process works by:
    1. Iterating over each character in the input string.
    2. Converting each character to its ASCII value using `ord()`, then formatting it as an 8-bit binary string.
    3. Concatenating the binary representations of all characters into a bitarray, and returning the binary string.
    """
    encoded = bitarray()
    for char in file_name:
        new_encoded = bitarray(format(ord(char), '08b'))
        encoded = encoded + new_encoded
    return encoded.to01()


def encode_content(content, file_name):
    """
    Encodes the given content into a compressed binary string using a combination of Elias Omega encoding,
    Huffman encoding, and LZ77 compression.

    Args:
        content (str): The input content (string) to be compressed and encoded.

    Returns:
        str: A binary string representing the encoded content, including the file header, Huffman encoding, 
             LZ77 compression, and necessary padding to align the output to a multiple of 8 bits.

    The encoding process works by:
    1. Encoding the header:
        - The length of the content is encoded using Elias Omega encoding.
        - The length of the file name is also encoded using Elias Omega encoding.
        - The file name is encoded using an 8-bit binary (ASCII) representation.
    2. Encoding the content:
        - A frequency dictionary is created from the content.
        - A Huffman tree is built from the frequency dictionary.
        - The Huffman tree and content are encoded using Huffman encoding.
        - The content is compressed using LZ77, and the compressed data is encoded using the Huffman tree.
    3. Adding padding:
        - Padding is added to ensure the final binary string is a multiple of 8 bits for proper byte alignment.
    """

    encoded = bitarray()

    # Encode the header
    encoded.extend(elias_omega_non_negative_encode(len(content)))
    encoded.extend(elias_omega_non_negative_encode(len(file_name)))
    encoded.extend(char_encode(file_name))

    # Encode the content

    freq_dict = character_frequency(content)
    huffman_tree = Huffman_Tree(freq_dict)
    encoded.extend(huffman_encode(huffman_tree, freq_dict))
    encoded.extend(lz77_encode(lz77_compress(content), huffman_tree))
    
    # Padding: Ensure the encoded bitstream is a multiple of 8 bits long
    padding_needed = (8 - len(encoded) % 8) % 8  # Calculate how many 0s are needed to pad to a multiple of 8
    if padding_needed > 0:
        encoded.extend('0' * padding_needed)  # Add the padding bits
    return encoded.to01()


def elias_omega_decode(encoded, current_index):
    """
    Decodes a value from the Elias Omega encoded string starting at the given index.

    Args:
        encoded (str): The Elias Omega encoded binary string.
        current_index (int): The index in the encoded string from which to start decoding.

    Returns:
        tuple: A tuple containing the decoded value (int) and the new index (int) 
               representing the position after the decoded value in the encoded string.

    The decoding process works by:
    1. Iteratively analyzing segments of the encoded string.
    2. For each segment:
        - If the first bit is '0', a new length is derived from the binary string by converting it into an integer.
        - If the first bit is '1', the decoded value is returned after converting the binary segment to an integer.
    3. The function returns the decoded integer and the index of the next bit after the decoded value.
    """

    end = current_index + 1

    while end <= len(encoded):
        # If the first bit of the current section is '0', decode the length of the next segment
        if encoded[current_index] == '0':
            # Create a new binary segment to decode the next length
            new_end = int('1' + encoded[current_index+1:end], 2) + 1
            current_index = end
            end = current_index + new_end
        else:
            # If the first bit is '1', decode the value from the binary segment
            return int(encoded[current_index:end], 2) - 1, end


def header_decode(encoded):
    """
    Decodes the header of a binary-encoded string, extracting content length, file name length, and file name.

    Args:
        encoded (str): The binary string containing the encoded header information.

    Returns:
        tuple: A tuple containing:
            - A dictionary (`header_info`) with the following keys:
                - 'content_length': The length of the content.
                - 'file_name_length': The length of the file name.
                - 'file_name': The actual file name extracted from the encoded string.
            - The index (`current_index`) where the header ends, and the remaining encoded content begins.

    The decoding process works by:
    1. Extracting and decoding the `content_length` and `file_name_length` using Elias Omega decoding.
    2. Using the `file_name_length` to extract the file name, decoding each 8-bit segment as an ASCII character.
    3. Returning the header information and the index of the next bit after the header.
    """

    header_info = {}
    current_index = 0

    keys = ['content_length', 'file_name_length']
    file_name = ''

    # Decode the content length and file name length using Elias Omega decoding
    for key in keys:
        decoded_value, new_start = elias_omega_decode(encoded, current_index)
        header_info[key] = (decoded_value)
        current_index = new_start

    # Decode the file name using the file_name_length
    for _ in range(header_info['file_name_length']):
        # Extract 8-bit segments, convert them to ASCII characters, and build the file name
        file_name += chr(int(encoded[current_index:current_index+8], 2))
        current_index += 8

    # Store the decoded file name in the header info
    header_info['file_name'] = file_name

    return header_info, current_index


def distinct_chars_decode(encoded, current_index):
    """
    Decodes the distinct characters and their associated bit patterns from the encoded binary string.

    Args:
        encoded (str): The binary string containing the encoded distinct characters and their corresponding bit patterns.
        current_index (int): The index in the encoded string from which to start decoding the distinct characters.

    Returns:
        tuple: A tuple containing:
            - A dictionary (`distinct_chars_info`) where:
                - The keys are distinct characters (decoded from 8-bit ASCII),
                - The values are the bit patterns associated with each character.
            - The new index (`new_index`) in the binary string after decoding the distinct characters and their bit patterns.

    The decoding process works by:
    1. Decoding the number of distinct characters using Elias Omega decoding.
    2. For each character:
        - Decoding the character from the next 8 bits (ASCII representation).
        - Decoding the length of the associated Huffman code using Elias Omega decoding.
        - Extracting the bit pattern of the specified length from the encoded string.
    3. Returning the dictionary of distinct characters and their bit patterns, along with the new index after decoding.
    """
    distinct_chars_info = {}
    num_distinct_chars, new_index = elias_omega_decode(encoded, current_index)
    for _ in range(num_distinct_chars):
        char = chr(int(encoded[new_index:new_index+8], 2))
        new_index += 8
        length, new_index = elias_omega_decode(encoded, new_index)
        distinct_chars_info[char] = (encoded[new_index:new_index+length])
        new_index += length
    return distinct_chars_info, new_index


def huffman_decode(distinct_chars_info, encoded_text):
    """
    Decodes a single character from the encoded text using the provided Huffman code mapping.

    Args:
        distinct_chars_info (dict): A dictionary where:
            - Keys are characters (str),
            - Values are their corresponding Huffman-encoded bit patterns (str).
        encoded_text (str): The binary string representing the Huffman-encoded text to decode.

    Returns:
        tuple: A tuple containing:
            - The decoded character (str),
            - The number of bits read (int) from the encoded text to decode the character.

    The decoding process works by:
    1. Accumulating bits from the `encoded_text` one by one.
    2. Comparing the accumulated bits against the Huffman code bit patterns in `distinct_chars_info`.
    3. Returning the corresponding character when a match is found, along with the number of bits consumed.

    Notes:
        - The function returns as soon as a matching Huffman code is found for a character.
        - The function expects that the encoded text follows the Huffman encoding, and assumes a well-formed input.
    """
    # Decode the encoded text using the Huffman tree
    accumulated_bits = ""
    counter = 0

    # Iterate over each bit in the encoded text
    for bit in encoded_text:
        accumulated_bits += bit # Accumulate bits one by one
        # Check if the accumulated bits match any Huffman code
        for key, value in distinct_chars_info.items():
            if accumulated_bits == value:
                # Return the decoded character and the number of bits consumed
                return key, counter + 1
        counter += 1    # Keep track of how many bits have been read


def lz77_decode(encoded_text, current_index, distinct_chars_info):
    """
    Decodes a given LZ77-encoded binary string starting from the specified index, using Elias Omega decoding
    for offsets and lengths, and Huffman decoding for characters.

    Args:
        encoded_text (str): The binary string containing the LZ77-encoded data.
        current_index (int): The index in the encoded string from where decoding should begin.
        distinct_chars_info (dict): A dictionary where:
            - Keys are distinct characters (str),
            - Values are their corresponding Huffman-encoded bit patterns (str).

    Returns:
        tuple: A tuple containing:
            - Decoded text (str): The decoded text after processing the LZ77-encoded binary data.
            - Padding (int or None): The number of padding bits (if any) at the end of the encoded text, 
                                     or `None` if there was no padding.

    The decoding process works by:
    1. Using Elias Omega decoding to decode the `offset` and `length` values in the LZ77 format.
    2. Using Huffman decoding to decode the next literal character.
    3. If the `offset` and `length` are both 0, the decoded character is directly added to the output.
    4. If `offset` and `length` are non-zero, the algorithm copies `length` characters from the position `offset` 
       back in the already decoded text and appends them to the output, followed by the decoded character.
    5. The process continues until the end of the encoded text or until padding (all 0s) is detected.
    """

    decoded_text = ""
    encoded = encoded_text[current_index:]
    new_index = 0
    padding = None
    while new_index < len(encoded):
        offset, new_index = elias_omega_decode(encoded, new_index)
        
        length, new_index = elias_omega_decode(encoded, new_index)
        char, counter  = huffman_decode(distinct_chars_info, encoded[new_index:])
        new_index += counter
        if offset == 0 and length == 0:
            decoded_text += char
        else:
            start_position = len(decoded_text) - offset
            for j in range(length):
                decoded_text += decoded_text[start_position + j]
                
            decoded_text += char
        if set(encoded[new_index:]) == {'0'}:
            padding = len(encoded[new_index:])
            # If the rest of the encoded text is all 0s, break because it's padding
            break
    return decoded_text, padding


def master_decode(encoded):
    """
    Decodes the entire encoded binary string using a combination of header decoding, Huffman decoding, 
    and LZ77 decompression.

    Args:
        encoded (str): The binary string containing the encoded data, including the header, 
                       distinct characters, and LZ77-encoded content.

    Returns:
        dict: A dictionary containing the decoded information with the following keys:
            - 'file_name' (str): The decoded file name from the header.
            - 'content_length' (int): The length of the original content.
            - 'distinct_chars' (list): A list of distinct characters that were Huffman-encoded.
            - 'decoded_text' (str): The fully decoded text after LZ77 decompression.
            - 'padding' (int or None): The number of padding bits, if any, in the encoded text.

    The decoding process works by:
    1. Decoding the header using the `header_decode` function to extract file name, content length, 
       and the starting index of the encoded content.
    2. Decoding the distinct characters and their Huffman-encoded bit patterns using the 
       `distinct_chars_decode` function.
    3. Decoding the LZ77-compressed content using the `lz77_decode` function.
    4. Collecting the distinct characters from the Huffman codes and returning all decoded 
       information in a dictionary.
    """

    header_info, current_index = header_decode(encoded)
    
    distinct_chars_info, new_index = distinct_chars_decode(encoded, current_index)
    decoded_text, padding = lz77_decode(encoded, new_index, distinct_chars_info)
    distinct_chars = []
    for key, _ in distinct_chars_info.items():
        distinct_chars.append(key)
    decoded = {
        'file_name': header_info['file_name'],
        'content_length': header_info['content_length'],
        'distinct_chars': distinct_chars,
        'decoded_text': decoded_text,
        'padding': padding
    }
    return decoded

if __name__ == "__main__":
    # Check if a file name was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python a2q2.py <filename>")
        sys.exit(1)

    # Get the file name from the command-line arguments
    file_name = sys.argv[1]

    try:
        # Open the file in read mode
        with open(file_name, 'r') as file:
            # Read the content of the file
            content = file.read()
        with open(f'{file_name}.bin', 'w') as f:
            f.write(encode_content(content, file_name))
        
        """
        Please uncomment the following codes to decode the encoded file
        (From line 665 to 675)
        """

        """
        Note: 
        - The encoded file will be saved as '<file_name>.bin' and the decoded file will be saved as '<file_name>_decoded.txt'
        - Please change the file_name in line 664 to your encoded file if you want to decode a different file.
            Otherwise, the program will decode the file that was encoded in the previous step.
        """
        
        # with open(f'{file_name}.bin', 'r') as read_file:
        #     stuff = read_file.read()

        # decodeed_file_name, content_length, distinct_chars, decoded_text, padding = master_decode(stuff).values()

        # with open(f'{decodeed_file_name}_decoded.txt', 'w') as write_file:
        #     write_file.write(f"File Name: {decodeed_file_name}\n")
        #     write_file.write(f"   Content Length: {content_length}\n")
        #     write_file.write(f"   Distinct Characters: {distinct_chars}\n")
        #     write_file.write(f"   Number of Zero as padding in the encoded file: {padding}\n")
        #     write_file.write(f" \nDecoded Content: \n\n{decoded_text}")

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
