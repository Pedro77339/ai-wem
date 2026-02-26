"""Allow running as: python -m ai_wem.demo"""

from ai_wem.demo.chat import main
import asyncio

asyncio.run(main())
