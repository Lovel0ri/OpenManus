import sys
import re
from datetime import datetime
from loguru import logger as _logger
from app.config import PROJECT_ROOT

# Default print and file log levels
_print_level = "INFO"
_logfile_level = "DEBUG"

# Error patterns and handlers
error_patterns = []

def handle_api_error(record):
    """处理 API 相关错误"""
    message = record['message']
    if "400" in message:
        print("\n=== API 400 错误自动诊断 ===")
        # print("可能原因:")
        # print("1. 请求格式错误")
        # print("2. 参数无效")
        # print("3. 模型不支持当前操作")
        # print("建议操作:")
        # print("- 检查 config.toml 中的模型配置")
        # print("- 确认 API 密钥格式正确")
        # print("- 查看 API 文档验证请求格式")
        # print("===========================\n")

# Add patterns to list
error_patterns.append((r"API.*error.*400", handle_api_error))


def _error_filter(record):
    """Loguru filter: 调用匹配模式的处理器"""
    # Only inspect error level and above
    if record['level'].no < 40:
        return True
    message = record['message']
    for pattern, handler in error_patterns:
        if re.search(pattern, message, flags=re.IGNORECASE):
            handler(record)
    return True


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """配置日志输出等级及格式"""
    global _print_level, _logfile_level
    _print_level = print_level
    _logfile_level = logfile_level

    # Generate timestamped log filename
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    log_name = f"{name}_{now}" if name else now
    log_path = PROJECT_ROOT / f"logs/{log_name}.log"

    # Remove default handlers
    _logger.remove()
    # Console stderr
    _logger.add(sys.stderr, level=_print_level, filter=_error_filter,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                       "<level>{message}</level>")
    # File
    _logger.add(str(log_path), level=_logfile_level, filter=_error_filter,
                rotation="10 MB", retention="30 days",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}")
    return _logger

# Initialize logger
logger = define_log_level()

if __name__ == "__main__":
    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
