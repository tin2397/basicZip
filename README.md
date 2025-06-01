# Text Compression Project

This project implements a lossless text compression scheme that combines LZ77 compression with Huffman coding and Elias-Omega encoding.

## Overview

The compression scheme uses three binary coding schemes:
1. Fixed-length 8-bit ASCII character code for characters
2. Variable-length Huffman code for characters
3. Variable-length Elias-Omega code for non-negative integers

## Requirements

- Python 3.x
- bitarray library (for binary I/O operations)

## Installation

```bash
pip install bitarray
```

## Usage

```bash
python a2q2.py <inputTextFileName>
```

The program will create a compressed binary file named `<inputTextFileName>.bin`

## Implementation Details

### Compression Process

The compression process consists of two main parts:

#### I. Header Part
1. **Metadata Encoding**
   - Size of input file (Elias code)
   - Length of filename (Elias code)
   - Filename (8-bit ASCII)

2. **Huffman Code Encoding**
   - Number of distinct characters (Elias code)
   - For each distinct character:
     - Character (8-bit ASCII)
     - Length of Huffman codeword (Elias code)
     - Huffman codeword itself

#### II. LZ77 Tuple Data Part
- Each tuple ⟨offset,length,next_char⟩ is encoded as:
  - Offset (Elias code)
  - Length (Elias code)
  - Next character (Huffman code)

### Output Format

The output is a binary stream of bits where:
- Each byte contains 8 consecutive bits
- The stream is padded with 0s if necessary to make the total length a multiple of 8

## Example

For an input file "x.asc" containing "aacaacabcaba":

1. LZ77 tuples generated:
   - ⟨0,0,a⟩
   - ⟨1,1,c⟩
   - ⟨3,4,b⟩
   - ⟨3,3,a⟩

2. The output file "x.asc.bin" will contain the binary stream of all encoded data.

## Notes

- The search window and lookahead buffer are considered infinitely long
- Elias-Omega code is adapted for non-negative integers by shifting by one
- The program handles padding automatically to ensure byte-aligned output

## Author

Quoc Hoang
