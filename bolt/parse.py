from optparse import *
import textwrap

loss_functions = [0,1,2,5,6]

class IndentedHelpFormatterWithNL(IndentedHelpFormatter):
  def format_description(self, description):
    if not description: return ""
    desc_width = self.width - self.current_indent
    indent = " "*self.current_indent
# the above is still the same
    bits = description.split('\n')
    formatted_bits = [
      textwrap.fill(bit,
        desc_width,
        initial_indent=indent,
        subsequent_indent=indent)
      for bit in bits]
    result = "\n".join(formatted_bits) + "\n"
    return result

  def format_option(self, option):
    # The help for each option consists of two parts:
    #   * the opt strings and metavars
    #   eg. ("-x", or "-fFILENAME, --file=FILENAME")
    #   * the user-supplied help string
    #   eg. ("turn on expert mode", "read data from FILENAME")
    #
    # If possible, we write both of these on the same line:
    #   -x    turn on expert mode
    #
    # But if the opt string list is too long, we put the help
    # string on a second line, indented to the same column it would
    # start in if it fit on the first line.
    #   -fFILENAME, --file=FILENAME
    #       read data from FILENAME
    result = []
    opts = self.option_strings[option]
    opt_width = self.help_position - self.current_indent - 2
    if len(opts) > opt_width:
      opts = "%*s%s\n" % (self.current_indent, "", opts)
      indent_first = self.help_position
    else: # start help on same line as opts
      opts = "%*s%-*s  " % (self.current_indent, "", opt_width, opts)
      indent_first = 0
    result.append(opts)
    if option.help:
      help_text = self.expand_default(option)
# Everything is the same up through here
      help_lines = []
      for para in help_text.split("\n"):
        help_lines.extend(textwrap.wrap(para, self.help_width))
# Everything is the same after here
      result.append("%*s%s\n" % (
        indent_first, "", help_lines[0]))
      result.extend(["%*s%s\n" % (self.help_position, "", line)
        for line in help_lines[1:]])
    elif opts[-1] != "\n":
      result.append("\n")
    return "".join(result)

def check_loss(option, opt_str, value, parser):
    if value not in loss_functions:
        raise OptionValueError("%d is not a valid loss function." % value)
    setattr(parser.values, option.dest, value)

def check_norm(option, opt_str, value, parser):
    if value not in [1,2]:
        raise OptionValueError("%d is not a valid penalty." % value)
    setattr(parser.values, option.dest, value)

def check_verbosity(option, opt_str, value, parser):
    if value not in range(3):
        raise OptionValueError("%d is not a valid verbosity level." % value)
    setattr(parser.values, option.dest, value)

def check_pos(option, opt_str, value, parser):
    if value <= 0.0:
        raise OptionValueError("%s must be larger than 0.0." % (opt_str, value))
    setattr(parser.values, option.dest, value)

def check_alpha(option,opt_str, value, parser):
    if value < 0.0 or value > 1.0:
        raise OptionValueError("alpha must be in [0,1]. ")
    setattr(parser.values, option.dest,value)

def parse(version):
    epilog = """the epilog."""
    parser = OptionParser(usage="%prog [options] example_file",
                          version="%prog "+version,
                          epilog=epilog,
                          formatter=IndentedHelpFormatterWithNL())
    parser.add_option("-v","--verbose", action="callback",
                      callback=check_verbosity,
                      dest="verbose",
                      help="verbose output",
                      default=1,
                      metavar="[0,1,2]",
                      type="int")
   
    parser.add_option("-l",
                      action="callback",
                      callback=check_loss,
                      help="Loss function to use. \n0: Hinge loss.\n"+
                      "1: Modified huber loss [default]. \n"+
                      "2: Log loss.\n"+"5: Squared loss.\n"+
                      "6: Huber loss.",
                      type="int",
                      dest="loss",
                      metavar="[0..]",
                      default=1)
    
    parser.add_option("-r",
                      dest="regularizer",
                      help="Regularization term lambda [default %default]. ",
                      type="float",
                      default=0.0001,
                      metavar="float")
    parser.add_option("-a","--alpha",
                      action="callback",
                      dest="alpha",
                      callback=check_alpha,
                      help="Elastic net hyper-parameter [default %default]. alpha < 1 is computationally intensive.",
                      type="float",
                      default=1.0,
                      metavar="float")
    parser.add_option("-e","--epsilon",
                      action="callback",
                      dest="epsilon",
                      callback=check_pos,
                      help="Size of the regression tube. ",
                      type="float",
                      metavar="float")
    parser.add_option("-n","--norm",
                      action="callback",
                      dest="norm",
                      callback=check_norm,
                      help="Penalty to use. \n1: L1 norm.\n"+
                      "2: L2 norm [default]. \n",
                      type="int",
                      metavar="[1,2]",
		      default = 2)
    parser.add_option("-E","--epochs", 
                      dest="epochs",
                      help="Number of epochs to perform [default %default]. ",
                      default=5,
                      metavar="int",
                      type="int")
    parser.add_option("--shuffle",
                      action="store_true",
                      dest="shuffle",
                      default=False,
                      help="Shuffle the training data after each epoche.")

    return parser

def parseSB(version):
    parser = parse(version)
    parser.add_option("-p", "--predictions",
                      dest="prediction_file",
                      help="Write predicitons to FILE. If FILE is '-' write to stdout [either -t or --test-only are required].",
                      metavar="FILE")
    parser.add_option("-t",
                      dest="test_file",
                      help="Evaluate the model on a seperate test file. ", 
                      metavar="FILE")
    parser.add_option("-m", "--model",
                      dest="model_file",
                      help="If --test-only: Apply seralized model in FILE to example_file. \nelse: store trained model in FILE.",
                      metavar = "FILE") 
    parser.add_option("--test-only",
                      action="store_true",
                      dest="test_only",
                      default=False,
                      help="Apply serialized model in option -m to example_file [requires -m].")
    return parser
    
  
def parseCV(version):
    parser = parse(version)
    parser.add_option("-f","--folds", 
		      dest="nfolds",
		      help="number of folds [default %default].",
		      default=10,
		      type="int",
		      metavar="int")
    return parser

