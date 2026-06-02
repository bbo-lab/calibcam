try:
    import reportlab
except ImportError:
    raise ImportError("The 'reportlab' library is required to run this script. Please install it using '(uv) pip install reportlab'.")

import os
import cv2

from reportlab.lib import pagesizes
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from calibcamlib import Board

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("board2pdf")

# Unit conversions
CM_TO_INCH = 1 / 2.54


def main(board_file: str, output_pdf: str = None, pagesize: str = "A4", dpi: int = 300):

    # Crate the board object from the file
    board = Board.from_file(board_file)
    board_params = board.get_board_params()

    square_size_real = board_params["square_size_real"]
    match board_params.get("unit", "cm"):
        case "cm":
            square_size_inch = square_size_real * CM_TO_INCH
        case "m":
            square_size_inch = square_size_real * 100 * CM_TO_INCH
        case _:
            raise ValueError(f"Unsupported unit: {board_params['unit']}")

    board_width_inch = board_params["boardWidth"] * square_size_inch
    board_height_inch = board_params["boardHeight"] * square_size_inch

    # Calculate the pixel dimensions with the target DPI
    board_width_dots = int(board_width_inch * dpi)
    board_height_dots = int(board_height_inch * dpi)

    # Calculate the board dimensions in points
    board_width_points = board_width_inch * inch
    board_height_points = board_height_inch * inch

    # Get the page size in points
    pagesize_dims = getattr(pagesizes, pagesize)
    page_width_points, page_height_points = pagesize_dims

    assert board_width_points <= page_width_points and board_height_points <= page_height_points, f"Board dimensions ({board_width_points} x {board_height_points} points) exceed page size ({page_width_points} x {page_height_points} points)"

    # Generate the board image at the required pixel dimensions
    board_img = board.get_board_img((board_width_dots, board_height_dots))

    # Save temporary image (ReportLab needs a file path)
    temp_img_path = "temp_charuco.png"
    cv2.imwrite(temp_img_path, board_img)

    # Creating the charuco PDF
    x_offset = (page_width_points - board_width_points) / 2
    y_offset = (page_height_points - board_height_points) / 2

    if output_pdf is None:
        board_name = board_params.get("board_name", 
                                      os.path.splitext(os.path.basename(board_file))[0])
        # Save the PDF in the same directory as the board file
        output_dir = os.path.dirname(board_file)
        output_pdf = output_dir + os.path.sep + f"{board_name}.pdf"

    c = canvas.Canvas(output_pdf, pagesize=pagesize_dims)

    c.drawImage(temp_img_path, x_offset, y_offset, width=board_width_points, height=board_height_points)
    c.showPage()
    c.save()

    # Clean up temporary image
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    logger.info(f"PDF saved to {output_pdf}")
    logger.info(f"Expected board dimensions: {board_width_inch / CM_TO_INCH:.2f} x {board_height_inch / CM_TO_INCH:.2f} cm on {pagesize} paper at {dpi} DPI.")


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description="Generate a PDF with the charuco board pattern.")
    parser.add_argument("board_file", type=str, help="Path to the .npy file containing the board parameters.")
    parser.add_argument("output_pdf", type=str, help="Path to the output PDF file.")
    parser.add_argument("--pagesize", type=str, default="A4", help="Page size for the PDF (e.g., A4, Letter).")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for the generated board image.")

    args = parser.parse_args()

    main(args.board_file, args.output_pdf, args.pagesize, args.dpi)