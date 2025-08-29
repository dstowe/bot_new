# auth/login_manager.py - SIMPLIFIED DID FROM CREDENTIALS ONLY

import time
import logging
import requests
import uuid
from typing import Tuple, Dict
from .credentials import CredentialManager

class LoginManager:
    """Handles login operations using DID from encrypted credentials only"""
    
    def __init__(self, wb, credential_manager: CredentialManager = None, logger=None):
        self.wb = wb
        self.credential_manager = credential_manager or CredentialManager()
        self.logger = logger or logging.getLogger(__name__)
        self.is_logged_in = False
        
        # Enhanced retry settings for image verification
        self.max_login_attempts = 5
        self.base_login_delay = 15
        self.image_verification_delay = 30
        self.max_trade_token_attempts = 3
        self.base_trade_token_delay = 10
    
    def _setup_did_from_credentials(self):
        """Setup DID from credentials only - ignores did.bin"""
        try:
            if not self.credential_manager.credentials_exist():
                self.logger.warning("⚠️  No credentials found - using auto-generated DID")
                self.logger.info("💡 Run credential setup to save DID permanently")
                # Use whatever DID webull auto-generated
                return False
            
            credentials = self.credential_manager.load_credentials()
            stored_did = credentials.get('did')
            
            if stored_did and len(stored_did) == 32:
                # Set DID directly on webull instance (bypass did.bin entirely)
                self.wb._did = stored_did
                
                # Also set it in headers
                if hasattr(self.wb, '_headers') and self.wb._headers:
                    self.wb._headers['did'] = stored_did
                
                self.logger.debug(f"✅ Using DID from credentials: {stored_did}")
                return True
            else:
                # No DID in credentials or invalid DID
                if stored_did:
                    self.logger.warning(f"⚠️  Invalid DID in credentials (length: {len(stored_did)})")
                else:
                    self.logger.warning("⚠️  No DID in credentials")
                
                # Generate new DID and try to save it
                new_did = uuid.uuid4().hex
                self.wb._did = new_did
                
                if hasattr(self.wb, '_headers') and self.wb._headers:
                    self.wb._headers['did'] = new_did
                
                # Try to save new DID to credentials
                try:
                    success = self.credential_manager.update_credentials(did=new_did)
                    if success:
                        self.logger.info(f"💾 Generated and saved new DID: {new_did}")
                    else:
                        self.logger.warning(f"⚠️  Generated DID but couldn't save: {new_did}")
                except Exception as e:
                    self.logger.warning(f"⚠️  Generated DID but save failed: {e}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error setting up DID: {e}")
            # Fall back to letting webull use its default DID
            return False
    
    def login_automatically(self) -> bool:
        """Automated login using DID from credentials only"""
        
        # Setup DID from credentials (ignore did.bin)
        self._setup_did_from_credentials()
        
        for attempt in range(1, self.max_login_attempts + 1):
            try:
                self.logger.info(f"🔑 Login attempt {attempt}/{self.max_login_attempts}...")
                
                # Load credentials
                credentials = self.credential_manager.load_credentials()
                
                # Validate credentials
                if not self.credential_manager.validate_credentials(credentials):
                    self.logger.error("❌ Invalid credentials found")
                    return False
                
                # Show which DID we're using
                current_did = getattr(self.wb, '_did', 'auto-generated')
                self.logger.debug(f"🔑 Using DID: {current_did}")
                
                # Add small random delay to avoid detection patterns
                base_delay = 2 + (attempt * 0.5)
                time.sleep(base_delay)
                
                # Login to Webull
                self.logger.info("📡 Contacting Webull servers...")
                login_result = self.wb.login(
                    username=credentials['username'],
                    password=credentials['password']
                )
                
                # Debug logging (remove this line if you don't want to see login results)
                self.logger.debug(f"🔍 Login result: {login_result}")
                
                # Check login result
                if 'accessToken' in login_result:
                    self.logger.info("✅ Webull login successful!")
                    self.is_logged_in = True
                    
                    # Try to get trade token with retries
                    if self._get_trade_token_with_retry(credentials['trading_pin']):
                        self.logger.info("🎉 Complete authentication successful!")
                        return True
                    else:
                        self.logger.error("❌ Failed to get trade token after retries")
                        
                else:
                    # Analyze login failure
                    error_msg = login_result.get('msg', 'Unknown error')
                    error_code = login_result.get('code', 'unknown')
                    
                    # Handle image verification specifically
                    if self._is_image_verification_error(login_result):
                        self.logger.warning(f"🖼️  Image verification triggered (attempt {attempt})")
                        self.logger.info(f"   Error: {error_msg}")
                        self.logger.info("   This is normal - Webull's anti-bot protection")
                        
                        # Use longer delay for image verification
                        if attempt < self.max_login_attempts:
                            delay = self.image_verification_delay
                            self.logger.info(f"⏳ Waiting {delay}s to let verification expire...")
                            time.sleep(delay)
                            continue
                    else:
                        self.logger.warning(f"❌ Login failed: {error_msg} (Code: {error_code})")
                        
                        # Check if this is a retryable error
                        if not self._is_retryable_login_error(login_result):
                            self.logger.error(f"❌ Non-retryable error: {error_msg}")
                            return False
                
            except Exception as e:
                self.logger.warning(f"❌ Login attempt {attempt} exception: {e}")
                
                # Check if this is a retryable exception
                if not self._is_retryable_exception(e):
                    self.logger.error(f"❌ Non-retryable exception: {e}")
                    return False
            
            # Calculate delay for next attempt (if not image verification)
            if attempt < self.max_login_attempts:
                delay = self.base_login_delay + (attempt * 5)
                delay = min(delay, 45)
                
                self.logger.info(f"⏳ Waiting {delay}s before retry {attempt + 1}...")
                time.sleep(delay)
        
        self.logger.error(f"❌ All {self.max_login_attempts} login attempts failed")
        return False
    
    def _is_image_verification_error(self, login_result: Dict) -> bool:
        """Check if the error is related to image verification"""
        error_code = login_result.get('code', '').lower()
        error_msg = login_result.get('msg', '').lower()
        
        image_verification_indicators = [
            'user.check.slider.pic.fail',
            'image verification failed',
            'slider verification',
            'captcha',
            'verification failed'
        ]
        
        for indicator in image_verification_indicators:
            if indicator in error_code or indicator in error_msg:
                return True
        
        return False
    
    def _get_trade_token_with_retry(self, trading_pin: str) -> bool:
        """Get trade token with retry logic"""
        for attempt in range(1, self.max_trade_token_attempts + 1):
            try:
                self.logger.info(f"🎫 Getting trade token (attempt {attempt}/{self.max_trade_token_attempts})...")
                
                if self.wb.get_trade_token(trading_pin):
                    self.logger.info("✅ Trade token obtained successfully")
                    return True
                else:
                    self.logger.warning(f"❌ Trade token attempt {attempt} failed")
                    
            except Exception as e:
                self.logger.warning(f"❌ Trade token attempt {attempt} exception: {e}")
            
            if attempt < self.max_trade_token_attempts:
                delay = self.base_trade_token_delay * attempt
                self.logger.info(f"⏳ Waiting {delay}s before trade token retry...")
                time.sleep(delay)
        
        self.logger.error(f"❌ Failed to get trade token after {self.max_trade_token_attempts} attempts")
        return False
    
    def _is_retryable_login_error(self, login_result: Dict) -> bool:
        """Determine if a login error is retryable"""
        error_code = login_result.get('code', '').lower()
        error_msg = login_result.get('msg', '').lower()
        
        if self._is_image_verification_error(login_result):
            return True
        
        non_retryable_codes = [
            'phone.illegal',
            'user.passwd.error',
            'account.freeze',
            'account.lock',
            'user.not.exist'
        ]
        
        non_retryable_messages = [
            'invalid username',
            'invalid password', 
            'account suspended',
            'account locked',
            'user not found'
        ]
        
        for code in non_retryable_codes:
            if code in error_code:
                return False
        
        for msg in non_retryable_messages:
            if msg in error_msg:
                return False
        
        return True
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Determine if an exception is retryable"""
        retryable_exceptions = [
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            ConnectionError,
            TimeoutError,
            OSError
        ]
        
        for exc_type in retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        return True
    
    # Keep the existing methods unchanged
    def login_with_credentials(self, username: str, password: str, trading_pin: str, 
                             device_name: str = '', mfa: str = '', 
                             question_id: str = '', question_answer: str = '') -> bool:
        """Login with explicit credentials"""
        # Setup DID from credentials first
        self._setup_did_from_credentials()
        
        try:
            self.logger.info("Attempting login with provided credentials...")
            
            login_result = self.wb.login(
                username=username,
                password=password,
                device_name=device_name,
                mfa=mfa,
                question_id=question_id,
                question_answer=question_answer
            )
            
            if 'accessToken' in login_result:
                self.logger.info("✅ Login successful")
                self.is_logged_in = True
                
                if self.wb.get_trade_token(trading_pin):
                    self.logger.info("✅ Trade token obtained")
                    return True
                else:
                    self.logger.error("❌ Failed to get trade token")
                    return False
            else:
                self.logger.error(f"❌ Login failed: {login_result.get('msg', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Exception during login: {e}")
            return False
    
    def logout(self) -> bool:
        """Logout from Webull"""
        try:
            if self.is_logged_in:
                response_code = self.wb.logout()
                if response_code == 200:
                    self.logger.info("🔐 Logged out successfully")
                    self.is_logged_in = False
                    return True
                else:
                    self.logger.warning(f"⚠️ Logout returned code: {response_code}")
                    return False
            else:
                self.logger.info("Already logged out")
                return True
        except Exception as e:
            self.logger.warning(f"⚠️ Logout warning: {e}")
            return False
    
    def check_login_status(self) -> bool:
        """Check if currently logged in"""
        try:
            current_account_id = self.wb._account_id
            current_zone = self.wb.zone_var
            
            try:
                test_quote = self.wb.get_quote('SPY')
                if test_quote and 'close' in test_quote:
                    self.is_logged_in = True
                    
                    self.wb._account_id = current_account_id
                    self.wb.zone_var = current_zone
                    
                    self.logger.debug(f"Login status verified, account context preserved: {current_account_id}")
                    return True
                else:
                    self.is_logged_in = False
                    return False
            except Exception as e:
                self.logger.debug(f"Login status check failed: {e}")
                self.is_logged_in = False
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking login status: {e}")
            self.is_logged_in = False
            return False
    
    def refresh_login(self) -> bool:
        """Refresh login disabled due to errors"""
        self.logger.info("Login refresh requested but disabled due to errors")
        self.is_logged_in = False
        return False
    
    def get_login_info(self) -> Dict:
        """Get information about current login status"""
        return {
            'is_logged_in': self.is_logged_in,
            'access_token_exists': bool(getattr(self.wb, '_access_token', None)),
            'trade_token_exists': bool(getattr(self.wb, '_trade_token', None)),
            'account_id': getattr(self.wb, '_account_id', None),
            'uuid': getattr(self.wb, '_uuid', None)
        }