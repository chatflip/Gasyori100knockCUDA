#include "TimerBase.h"

std::string TimerBase::createHeader(const std::string& testName) const {
  std::ostringstream header;
  header << testName << std::endl;

  std::string buildType = "";
#if _DEBUG
  header << "Build: Debug" << std::endl;
#else
  header << "Build: Release" << std::endl;
#endif
  return header.str();
};

std::string TimerBase::createFooter(const float elapsedTime) const {
  std::ostringstream footer;
  footer << std::fixed << std::setprecision(2);
  footer << "Elapsed time: " << elapsedTime << " ms" << std::endl;
  return footer.str();
};