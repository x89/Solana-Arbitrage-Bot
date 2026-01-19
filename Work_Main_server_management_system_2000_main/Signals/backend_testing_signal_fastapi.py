#!/usr/bin/env python3
"""
FastAPI Signal Backend Server
Receives and processes trading signals from the test version signal generator
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
import json
import logging
from datetime import datetime
import os
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_backend_fastapi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Signal Backend API",
    description="FastAPI backend for receiving and processing trading signals",
    version="1.0.0"
)

class SignalAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    EXIT_BUY = "exit_buy"
    EXIT_SELL = "exit_sell"
    PNL_STATUS = "pnl_status"
    STATS = "stats"  # Also accept "stats" action

class BaseSignal(BaseModel):
    ticker: str = Field(..., description="Trading pair ticker")
    action: SignalAction = Field(..., description="Signal action type")
    price: str = Field(..., description="Signal price")
    time: str = Field(..., description="Signal timestamp")
    TotalTrades: str = Field(..., description="Total trades counter")

    @validator('ticker')
    def validate_ticker(cls, v):
        if v != "SOLUSDT.P":
            raise ValueError('Ticker must be "SOLUSDT.P"')
        return v

    @validator('price')
    def validate_price(cls, v):
        try:
            price = float(v)
            if price <= 0:
                raise ValueError('Price must be greater than 0')
        except (ValueError, TypeError):
            raise ValueError('Price must be a valid positive number')
        return v

    @validator('TotalTrades')
    def validate_total_trades(cls, v):
        try:
            trades = int(v)
            if trades <= 0:
                raise ValueError('TotalTrades must be greater than 0')
        except (ValueError, TypeError):
            raise ValueError('TotalTrades must be a valid positive integer')
        return v

class ExitSignal(BaseSignal):
    sl: str = Field(..., description="Stop loss")
    tp: str = Field(..., description="Take profit")
    per: str = Field(..., description="Percentage")

    @validator('action')
    def validate_exit_action(cls, v):
        if v not in [SignalAction.EXIT_BUY, SignalAction.EXIT_SELL]:
            raise ValueError('Exit signals must have action "exit_buy" or "exit_sell"')
        return v

class PNLStatusSignal(BaseModel):
    ticker: str = Field(..., description="Trading pair ticker")
    action: SignalAction = Field(..., description="Signal action type")
    time: str = Field(..., description="Signal timestamp")
    PNL: str = Field(..., description="Current PNL")
    LastPNL: str = Field(..., description="Cumulative PNL")
    MaxDraw: str = Field(..., description="Maximum drawdown")
    WinRate: str = Field(..., description="Win rate percentage")
    DailyPNL: str = Field(..., description="Daily PNL")
    Strategy: str = Field(..., description="Strategy identifier")
    StartDate: str = Field(..., description="Start date")
    historyPNL: str = Field(..., description="History PNL")
    TotalTrades: str = Field(..., description="Total trades")

    @validator('action')
    def validate_pnl_action(cls, v):
        if v not in [SignalAction.PNL_STATUS, SignalAction.STATS]:
            raise ValueError('PNL status signals must have action "pnl_status" or "stats"')
        return v

    @validator('ticker')
    def validate_ticker(cls, v):
        if v != "SOLUSDT.P":
            raise ValueError('Ticker must be "SOLUSDT.P"')
        return v

class SignalProcessor:
    def __init__(self):
        self.received_signals = []
        self.signal_stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'exit_buy_signals': 0,
            'exit_sell_signals': 0,
            'last_signal_time': None
        }
        
        # Create data directory
        os.makedirs('backend_signals_fastapi', exist_ok=True)
    
    def process_signal(self, signal_data: Dict) -> Dict:
        """Process incoming signal and return response"""
        try:
            # Determine signal type and validate
            action = signal_data.get('action')
            
            if action in ['pnl_status', 'stats']:
                # Validate PNL status signal
                pnl_signal = PNLStatusSignal(**signal_data)
                signal_dict = pnl_signal.dict()
            elif action in ['exit_buy', 'exit_sell']:
                # Validate exit signal
                exit_signal = ExitSignal(**signal_data)
                signal_dict = exit_signal.dict()
            else:
                # Validate base signal
                base_signal = BaseSignal(**signal_data)
                signal_dict = base_signal.dict()
            
            # Signal is valid - process it
            self.received_signals.append(signal_dict)
            self.signal_stats['total_signals'] += 1
            self.signal_stats['last_signal_time'] = datetime.now().isoformat()
            
            # Update action-specific stats
            if action == 'buy':
                self.signal_stats['buy_signals'] += 1
            elif action == 'sell':
                self.signal_stats['sell_signals'] += 1
            elif action == 'exit_buy':
                self.signal_stats['exit_buy_signals'] += 1
            elif action == 'exit_sell':
                self.signal_stats['exit_sell_signals'] += 1
            elif action in ['pnl_status', 'stats']:
                # Track PNL status signals
                if 'pnl_status_signals' not in self.signal_stats:
                    self.signal_stats['pnl_status_signals'] = 0
                self.signal_stats['pnl_status_signals'] += 1
            
            # Save signal to file
            self.save_signal(signal_dict)
            
            # Log the signal
            if action in ['pnl_status', 'stats']:
                pnl_value = signal_dict.get('PNL', 'N/A')
                total_trades = signal_dict.get('TotalTrades', 'N/A')
                logger.info(f"PNL status processed successfully: PNL={pnl_value}, TotalTrades={total_trades}")
            else:
                price = signal_dict.get('price', 'N/A')
                total_trades = signal_dict.get('TotalTrades', 'N/A')
                logger.info(f"Signal processed successfully: {action} at ${price} (Trade #{total_trades})")
            
            return {
                'status': 'success',
                'message': f'Signal processed: {action}',
                'signal_id': len(self.received_signals),
                'received_signal': signal_dict
            }
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return {
                'status': 'error',
                'message': f'Validation error: {str(e)}',
                'received_signal': signal_data
            }
    
    def save_signal(self, signal: Dict):
        """Save signal to daily file"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            filename = f"backend_signals_fastapi/signals_{today}.json"
            
            # Read existing signals
            try:
                with open(filename, 'r') as f:
                    signals = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                signals = []
            
            # Add timestamp
            signal_with_timestamp = {
                'received_at': datetime.now().isoformat(),
                'signal': signal
            }
            
            # Add new signal
            signals.append(signal_with_timestamp)
            
            # Save back to file
            with open(filename, 'w') as f:
                json.dump(signals, f, indent=2)
            
            logger.info(f"Signal saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'stats': self.signal_stats,
            'total_received': len(self.received_signals),
            'last_10_signals': self.received_signals[-10:] if self.received_signals else []
        }
    
    def get_all_signals(self) -> List[Dict]:
        """Get all received signals"""
        return self.received_signals

# Initialize signal processor
signal_processor = SignalProcessor()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Signal Backend API",
        "version": "1.0.0",
        "endpoints": {
            "POST /signal": "Receive trading signal",
            "GET /health": "Health check",
            "GET /stats": "Get processing statistics",
            "GET /signals": "Get all signals",
            "GET /signals/{id}": "Get specific signal",
            "POST /clear": "Clear all signals",
            "GET /docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'server': 'FastAPI Signal Backend'
    }

@app.post("/signal")
async def receive_signal(request: Request):
    """Receive and process trading signal"""
    try:
        # Get signal data
        signal_data = await request.json()
        
        if not signal_data:
            raise HTTPException(status_code=400, detail="No signal data provided")
        
        logger.info(f"Received signal: {signal_data}")
        
        # Process the signal
        result = signal_processor.process_signal(signal_data)
        
        # Return appropriate response
        if result['status'] == 'success':
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)
            
    except Exception as e:
        logger.error(f"Error handling request: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Alias endpoint to accept same payload at /account_server
@app.post("/account_server")
async def receive_signal_account_server(request: Request):
    try:
        signal_data = await request.json()
        if not signal_data:
            raise HTTPException(status_code=400, detail="No signal data provided")
        logger.info(f"Received (alias) signal: {signal_data}")
        result = signal_processor.process_signal(signal_data)
        if result['status'] == 'success':
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)
    except Exception as e:
        logger.error(f"Error handling request (/account_server): {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get signal processing statistics"""
    return signal_processor.get_stats()

@app.get("/signals")
async def get_signals():
    """Get all received signals"""
    return {
        'signals': signal_processor.get_all_signals(),
        'total': len(signal_processor.get_all_signals())
    }

@app.get("/signals/{signal_id}")
async def get_signal(signal_id: int):
    """Get specific signal by ID"""
    signals = signal_processor.get_all_signals()
    
    if signal_id < 1 or signal_id > len(signals):
        raise HTTPException(
            status_code=404, 
            detail=f"Signal ID {signal_id} not found. Available IDs: 1-{len(signals)}"
        )
    
    return {
        'signal': signals[signal_id - 1],
        'signal_id': signal_id
    }

@app.post("/clear")
async def clear_signals():
    """Clear all stored signals (for testing)"""
    signal_processor.received_signals.clear()
    signal_processor.signal_stats = {
        'total_signals': 0,
        'buy_signals': 0,
        'sell_signals': 0,
        'exit_buy_signals': 0,
        'exit_sell_signals': 0,
        'last_signal_time': None
    }
    
    logger.info("All signals cleared")
    return {
        'status': 'success',
        'message': 'All signals cleared'
    }

if __name__ == "__main__":
    import uvicorn
    
    print("FastAPI Signal Backend Server")
    print("=" * 50)
    print("Endpoints:")
    print("  POST /signal     - Receive trading signal")
    print("  GET  /health     - Health check")
    print("  GET  /stats      - Get processing statistics")
    print("  GET  /signals    - Get all signals")
    print("  GET  /signals/N  - Get specific signal")
    print("  POST /clear      - Clear all signals")
    print("  GET  /docs       - Interactive API documentation")
    print()
    print("Starting server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 