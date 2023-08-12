import argparse

def main():
    # Creating an ArgumentParser object and provide a description for the script
    parser= argparse.ArgumentParser(description=" A simple script with command line arguments")
    # defining command-line arguments using the add_argument() method
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', default='output.txt', help='Output file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose')

    args= parser.parse_args()

    if args.verbose:
        print("Verbose model enbaled")

    # print the value of the pasrsed argument
    print(f"Input file: {args.input}")
    print(f"Output files: {args.output}")


if __name__=='__main__':
    main()
