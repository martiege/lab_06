#include "lab_6.h"
#include <iostream>

int main()
{
  try
  {
    lab6();
  }
  catch (const std::exception& e)
  {
    std::cerr << "Caught exception:\n"
              << e.what() << '\n';
    return EXIT_FAILURE; 
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception\n";
    return EXIT_FAILURE; 
  }

  return EXIT_SUCCESS;
}
