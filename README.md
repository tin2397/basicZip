# Huffman and LZ77 Compression Implementation

This project implements a combination of Huffman coding and LZ77 compression algorithms for text compression. The implementation includes both encoding and decoding functionality.

## Features

- Huffman coding for character frequency-based compression
- LZ77 compression for pattern matching
- Elias Omega encoding for efficient number representation
- Support for file compression and decompression
- Detailed documentation and comments

## Requirements

- Python 3.x
- bitarray package

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
```

2. Install the required package:
```bash
pip install bitarray
```

## Usage

To compress a file:
```bash
python a2q2.py <filename>
```

The program will create a compressed binary file named `<filename>.bin`.

To decompress the file, uncomment the decoding section in the code and run the program again.

## Implementation Details

The compression process involves:
1. Building a Huffman tree based on character frequencies
2. Encoding the Huffman tree structure
3. Applying LZ77 compression to the input text
4. Combining both compression methods for optimal results

The decompression process reverses these steps to reconstruct the original text.

## License

[Your chosen license] 