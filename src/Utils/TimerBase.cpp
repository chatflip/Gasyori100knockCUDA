#include "TimerBase.h"

std::string TimerBase::createHeader(const std::string& testName) const {
  std::ostringstream header;
  header << testName << std::endl;
  std::string cpuName = getWMIInfo("Win32_Processor", "Name");
  std::string gpuName = getWMIInfo("Win32_VideoController", "Name");
  header << "CPU: " << cpuName << std::endl;
  header << "GPU: " << gpuName << std::endl;

  std::string buildType = "";
#if _DEBUG
  header << "Build: Debug" << std::endl;
#else
  header << "Build: Release" << std::endl;
#endif
  header << "-----------------------------------------\n";
  return header.str();
};

std::string TimerBase::createFooter(const float elapsedTime) const {
  std::ostringstream footer;
  footer << "-----------------------------------------\n";
  footer << std::fixed << std::setprecision(2);
  footer << "Elapsed time: " << elapsedTime << " ms" << std::endl;
  return footer.str();
};

void TimerBase::print(const std::string& header,
                      const std::string& footer) const {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2) << header;
  for (auto& it : records) {
    ss << it.first << " " << it.second << " ms" << std::endl;
  }
  ss << footer;
  std::cout << ss.str();
}

void TimerBase::writeToFile(const std::string& path, const std::string& header,
                            const std::string& footer) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    std::cerr << "Error: could not open file " << path << std::endl;
    return;
  }
  ofs << std::fixed << std::setprecision(2) << header;
  for (auto& it : records) {
    ofs << it.first << " " << it.second << " ms" << std::endl;
  }
  ofs << footer;
  ofs.close();
}

float TimerBase::getRecord(const std::string& name) const {
  if (records.count(name) == 0) {
    std::cerr << "Timer Error: " << name << " has not been recorded."
              << std::endl;
    return -1.0f;
  }
  return records.at(name);
}

void TimerBase::popRecord(const std::string& name) {
  if (records.count(name) == 0) {
    std::cerr << "Timer Error: " << name << " has not been recorded."
              << std::endl;
    return;
  }
  records.erase(name);
}

void TimerBase::mergeRecords(TimerBase& other) {
  for (auto& it : other.records) {
    records[it.first] = it.second;
  }
  other.records.clear();
}

std::string TimerBase::getWMIInfo(const std::string& wmiClass,
                                  const std::string& property) const {
  std::string result;
  HRESULT hres;

  hres = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  if (FAILED(hres)) {
    return result;
  }

  hres = CoInitializeSecurity(
      nullptr, -1, nullptr, nullptr, RPC_C_AUTHN_LEVEL_DEFAULT,
      RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE, nullptr);
  if (FAILED(hres)) {
    CoUninitialize();
    return result;
  }

  IWbemLocator* pLoc = nullptr;
  hres = CoCreateInstance(CLSID_WbemLocator, nullptr, CLSCTX_INPROC_SERVER,
                          IID_IWbemLocator, reinterpret_cast<LPVOID*>(&pLoc));
  if (FAILED(hres)) {
    CoUninitialize();
    return result;
  }

  IWbemServices* pSvc = nullptr;
  hres = pLoc->ConnectServer(_bstr_t(L"ROOT\\CIMV2"), nullptr, nullptr, nullptr,
                             0, nullptr, nullptr, &pSvc);
  if (FAILED(hres)) {
    pLoc->Release();
    CoUninitialize();
    return result;
  }

  hres = CoSetProxyBlanket(pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr,
                           RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE,
                           nullptr, EOAC_NONE);
  if (FAILED(hres)) {
    pSvc->Release();
    pLoc->Release();
    CoUninitialize();
    return result;
  }

  IEnumWbemClassObject* pEnumerator = nullptr;
  hres = pSvc->ExecQuery(
      bstr_t("WQL"),
      bstr_t(("SELECT " + property + " FROM " + wmiClass).c_str()),
      WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, nullptr,
      &pEnumerator);
  if (FAILED(hres)) {
    pSvc->Release();
    pLoc->Release();
    CoUninitialize();
    return result;
  }

  IWbemClassObject* pclsObj = nullptr;
  ULONG uReturn = 0;
  while (pEnumerator) {
    hres = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);
    if (0 == uReturn) {
      break;
    }

    VARIANT vtProp;
    hres = pclsObj->Get(bstr_t(property.c_str()), 0, &vtProp, nullptr, nullptr);
    if (SUCCEEDED(hres)) {
      if (vtProp.vt == VT_BSTR) {
        result = _bstr_t(vtProp.bstrVal);
      }
      VariantClear(&vtProp);
    }
    pclsObj->Release();
  }

  pEnumerator->Release();
  pSvc->Release();
  pLoc->Release();
  CoUninitialize();

  return result;
}