def remove_comments(inputFile,outputFile):
    '''
    Args:
        inputFile  (str): path of the file to remove commments from.
        outputFile (str): paths of the file to save after removing comments.

    Returns:
        bool: The return value. True for success, False otherwise.
    '''

    with open(inputFile, 'r') as f:
        lines = f.readlines()

    with open(outputFile, 'w') as f:
        for line in lines:
            # Keep the Shebang line
            if line[0:2] == "#!":
                f.writelines(line)
            # Also keep existing empty lines
            elif not line.strip():
                f.writelines(line)
            # But remove comments from other lines
            else:
                line = line.split('#')
                stripped_string = line[0].rstrip()
                # Write the line only if the comment was after the code.
                # Discard lines that only contain comments.
                if stripped_string:
                    f.writelines(stripped_string)
                    f.writelines('\n')
    return True


if __name__ == '__main__':
    remove_comments('detection.py', 'detection_no_comments.py')
