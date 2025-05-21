import copy
import math
from datetime import datetime
import numpy as np
from .strategy import Strategy
from ..log_manager import LogManager
from ..date_converter import DateConverter
from decimal import Decimal, ROUND_DOWN

class StrategyRsiAdvanced(Strategy):
    """
    고급 RSI(Relative Strength Index) 매매 전략:
    
    - 기본 RSI 매매 신호에 기반
    - 볼린저 밴드 통합하여 매매 시그널 강화
    - 변동성 기반 손절매/익절 로직 추가
    - 시장 상태(추세/횡보장)에 따라 매매 비중 조절
    - 동적 RSI 기준점 적용 (변동성에 따라 조정)
    - 부분적 익절 기능 추가
    - 이동평균선 기반 추세 확인 로직 추가
    - 초기 진입 최적화를 위한 추가 필터 구현
    
    Advanced RSI Trading Strategy:
    
    - Based on basic RSI trading signals
    - Integrates Bollinger Bands to strengthen trading signals
    - Added volatility-based stop-loss/take-profit logic
    - Adjusts trading volume based on market condition (trend/range)
    - Applies dynamic RSI thresholds (adjusted according to volatility)
    - Added partial take-profit functionality
    - Added moving average trend confirmation logic
    - Implemented additional filters for optimized entry points
    """
    
    ISO_DATEFORMAT = "%Y-%m-%dT%H:%M:%S"
    COMMISSION_RATIO = 0.0005
    
    # RSI 관련 설정
    RSI_BASE_LOW = 30  # 기본 RSI 매수 기준점
    RSI_BASE_HIGH = 70  # 기본 RSI 매도 기준점
    RSI_COUNT = 14  # RSI 계산 기간
    
    # 변동성 관련 설정
    VOLATILITY_PERIOD = 20  # 변동성 계산 기간
    VOLATILITY_WEIGHT = 0.3  # 변동성 반영 가중치
    
    # 손절/익절 관련 설정
    STOP_LOSS_RATIO = 0.03  # 기본 손절매 비율 (3%)
    TAKE_PROFIT_RATIO = 0.05  # 기본 익절 비율 (5%)
    
    # 볼린저 밴드 관련 설정
    BB_PERIOD = 20  # 볼린저 밴드 계산 기간
    BB_STD_DEV = 2.0  # 볼린저 밴드 표준편차 배수
    
    # 이동평균선 관련 설정
    MA_SHORT_PERIOD = 10  # 단기 이동평균 기간
    MA_MEDIUM_PERIOD = 20  # 중기 이동평균 기간
    MA_LONG_PERIOD = 50  # 장기 이동평균 기간
    
    # 부분 익절 관련 설정
    PARTIAL_TP_RATIO = 0.5  # 부분 익절 비율 (50%)
    PARTIAL_TP_TRIGGER = 0.03  # 부분 익절 트리거 (수익률 3% 이상)
    
    NAME = "RSI Advanced"
    CODE = "RSI-A"
    
    def __init__(self):
        self.is_intialized = False
        self.is_simulation = False
        self.rsi_info = None
        self.rsi = []
        self.data = []
        self.result = []
        self.add_spot_callback = None
        self.add_line_callback = None
        self.alert_callback = None
        self.budget = 0
        self.balance = 0
        self.asset_amount = 0
        self.min_price = 0
        self.logger = LogManager.get_logger(__class__.__name__)
        self.waiting_requests = {}
        self.position = None
        
        # 시장 상태 및 위험 관리 변수
        self.volatility = None  # 현재 변동성
        self.market_condition = "unknown"  # 시장 상태 (trend, range, unknown)
        self.entry_price = None  # 진입 가격
        self.rsi_dynamic_low = self.RSI_BASE_LOW  # 동적 RSI 매수 기준점
        self.rsi_dynamic_high = self.RSI_BASE_HIGH  # 동적 RSI 매도 기준점
        self.risk_ratio = 1.0  # 리스크 조절 비율 (0.1~1.0)
        
        # ATR 관련 변수
        self.atr_period = 14  # ATR 계산 기간
        self.atr_values = []  # ATR 계산을 위한 True Range 값들
        self.current_atr = None  # 계산된 현재 ATR 값
        self.sl_atr_factor = 1.5  # 손절매에 사용할 ATR 배수
        self.tp_atr_factor = 2.0  # 익절에 사용할 ATR 배수
        
        # 손절매/익절 가격 기록
        self.stop_loss_price = None
        self.take_profit_price = None
        self.partial_take_profit_price = None
        self.partial_tp_executed = False
        
        # 볼린저 밴드 관련 변수
        self.bb_upper = None
        self.bb_middle = None
        self.bb_lower = None
        
        # 이동평균선 관련 변수
        self.ma_short = None
        self.ma_medium = None
        self.ma_long = None
        self.ma_trend = "neutral"  # 이동평균선 추세 상태
        
        # 거래 이력 관련 변수
        self.last_trade_result = None
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3  # 연속 손실 허용 최대값
        self.cool_down_period = 6  # 손실 후 대기 기간 (캔들 수)
        self.cool_down_counter = 0
    
    def initialize(
        self,
        budget,
        min_price=500,  # 최소 거래 금액
        add_spot_callback=None,
        add_line_callback=None,
        alert_callback=None,
        atr_period=14,
        sl_atr_factor=1.5,
        tp_atr_factor=2.0,
        bb_period=20,
        bb_std_dev=2.0,
        ma_short_period=10,
        ma_medium_period=20,
        ma_long_period=50
    ):
        """
        전략 초기화
        
        Args:
            budget: 예산
            min_price: 최소 거래 금액, 거래소의 최소 거래 금액
            add_spot_callback: 그래프에 그려질 spot을 추가하는 콜백 함수
            add_line_callback: 그래프에 그려질 line을 추가하는 콜백 함수
            alert_callback: 알림을 전달하는 콜백 함수
            atr_period: ATR 계산 기간
            sl_atr_factor: 손절매에 사용할 ATR 배수
            tp_atr_factor: 익절에 사용할 ATR 배수
            bb_period: 볼린저 밴드 계산 기간
            bb_std_dev: 볼린저 밴드 표준편차 배수
            ma_short_period: 단기 이동평균 기간
            ma_medium_period: 중기 이동평균 기간
            ma_long_period: 장기 이동평균 기간
        """
        if self.is_intialized:
            return
        
        self.is_intialized = True
        self.budget = budget
        self.balance = budget
        self.min_price = min_price
        self.add_spot_callback = add_spot_callback
        self.add_line_callback = add_line_callback
        self.alert_callback = alert_callback
        
        # 기술적 지표 설정
        self.atr_period = atr_period
        self.sl_atr_factor = sl_atr_factor
        self.tp_atr_factor = tp_atr_factor
        self.BB_PERIOD = bb_period
        self.BB_STD_DEV = bb_std_dev
        self.MA_SHORT_PERIOD = ma_short_period
        self.MA_MEDIUM_PERIOD = ma_medium_period
        self.MA_LONG_PERIOD = ma_long_period
        
        self.logger.info(f"[RSI-A] Strategy initialized with budget: {budget}")
    
    def set_simulation_mode(self, is_simulation):
        """시뮬레이션 모드 설정"""
        self.is_simulation = is_simulation
    
    def get_request(self):
        """
        현재 상태에 따라 거래 요청 정보를 생성
        
        Returns:
            list: 요청 정보 리스트 또는 None
        """
        if not self.is_intialized or len(self.data) == 0:
            return None
        
        try:
            last_data = self.data[-1]
            now = datetime.now().strftime(self.ISO_DATEFORMAT)
            if self.is_simulation:
                now = last_data["date_time"]
            
            if last_data is None or self.position is None:
                if self.is_simulation:
                    return [{
                        "id": DateConverter.timestamp_id(),
                        "type": "buy",
                        "price": 0,
                        "amount": 0,
                        "date_time": now,
                    }]
                return None
            
            # 현재 가격 확인
            current_price = float(last_data["closing_price"])
            
            # 쿨다운 카운터 감소
            if self.cool_down_counter > 0:
                self.cool_down_counter -= 1
                if self.cool_down_counter == 0 and self.alert_callback is not None:
                    self.alert_callback("쿨다운 기간이 끝났습니다. 정상 매매를 재개합니다.")
            
            # 쿨다운 기간 중에는 새로운 매매 중단
            if self.cool_down_counter > 0:
                if self.position in ["buy", "sell"]:
                    self.position = None
            
            # 부분 익절 조건 확인
            if self.asset_amount > 0 and not self.partial_tp_executed and self.entry_price is not None:
                current_profit_ratio = (current_price / self.entry_price) - 1
                
                # 수익률이 부분 익절 트리거보다 높을 때
                if current_profit_ratio >= self.PARTIAL_TP_TRIGGER:
                    # 부분 익절 실행
                    self.logger.info(f"[PARTIAL TP] Triggered at {current_price:.2f} (Profit: {current_profit_ratio*100:.2f}%)")
                    if self.alert_callback is not None:
                        self.alert_callback(f"부분 익절 실행! 수익률 {current_profit_ratio*100:.2f}%")
                    self.partial_tp_executed = True
                    return self.__create_partial_take_profit(current_price)
            
            # 손절매/익절 조건 확인
            if self.asset_amount > 0 and self.stop_loss_price is not None and self.take_profit_price is not None:
                # 손절매 조건
                if current_price <= self.stop_loss_price:
                    self.logger.info(f"[STOP LOSS] Triggered at {current_price:.2f} (SL price: {self.stop_loss_price:.2f})")
                    if self.alert_callback is not None:
                        self.alert_callback(f"손절매 실행! 가격 {current_price:.2f}")
                    
                    # 손실 카운터 증가 및 쿨다운 설정
                    self.consecutive_losses += 1
                    if self.consecutive_losses >= self.max_consecutive_losses:
                        self.cool_down_counter = self.cool_down_period
                        if self.alert_callback is not None:
                            self.alert_callback(f"연속 {self.consecutive_losses}번 손절로 {self.cool_down_period}주기 동안 매매 중단")
                    
                    # 손절매 실행 후 상태 초기화
                    self.stop_loss_price = None
                    self.take_profit_price = None
                    self.partial_tp_executed = False
                    return self.__create_sell(current_price, self.asset_amount)
                
                # 익절 조건
                if current_price >= self.take_profit_price:
                    self.logger.info(f"[TAKE PROFIT] Triggered at {current_price:.2f} (TP price: {self.take_profit_price:.2f})")
                    if self.alert_callback is not None:
                        self.alert_callback(f"익절 실행! 가격 {current_price:.2f}")
                    
                    # 익절 성공 시 연속 손실 카운터 초기화
                    self.consecutive_losses = 0
                    
                    # 익절 실행 후 상태 초기화
                    self.stop_loss_price = None
                    self.take_profit_price = None
                    self.partial_tp_executed = False
                    return self.__create_sell(current_price, self.asset_amount)
            
            # 쿨다운 기간 중이면 매매 중단
            if self.cool_down_counter > 0:
                return None
            
            request = None
            
            # 매수/매도 요청 생성
            if self.position == "buy":
                # 추가 매매 필터 체크
                if self._is_good_entry_point():
                    # 종가로 최대 매수 (리스크 비율 적용)
                    request = self.__create_buy(current_price)
                else:
                    self.logger.info(f"[FILTER] Buy signal filtered out by additional conditions")
            elif self.position == "sell":
                # 종가로 전량 매도
                request = self.__create_sell(current_price, self.asset_amount)
            
            if request is None:
                if self.is_simulation:
                    return [{
                        "id": DateConverter.timestamp_id(),
                        "type": "buy",
                        "price": 0,
                        "amount": 0,
                        "date_time": now,
                    }]
                return None
            
            request["date_time"] = now
            
            # 로그 기록
            self.logger.info(f"[REQ] id: {request['id']} : {request['type']} ==============")
            self.logger.info(f"price: {request['price']}, amount: {request['amount']}")
            self.logger.info(f"market: {self.market_condition}, risk: {self.risk_ratio:.2f}")
            if self.bb_lower is not None and self.bb_upper is not None:
                self.logger.info(f"BB: lower={self.bb_lower:.2f}, upper={self.bb_upper:.2f}")
            self.logger.info(f"MA trend: {self.ma_trend}")
            self.logger.info("================================================")
            
            # 결과 반환
            final_requests = []
            for request_id in self.waiting_requests:
                self.logger.info(f"cancel request added! {request_id}")
                final_requests.append({
                    "id": request_id,
                    "type": "cancel",
                    "price": 0,
                    "amount": 0,
                    "date_time": now,
                })
            final_requests.append(request)
            return final_requests
        
        except (ValueError, KeyError) as msg:
            self.logger.error(f"invalid data {msg}")
        except IndexError:
            self.logger.error("empty data")
        except AttributeError as msg:
            self.logger.error(f"attribute error: {msg}")
        except Exception as e:
            self.logger.error(f"unexpected error in get_request: {e}")
        return None
    
    def update_trading_info(self, info):
        """
        새로운 거래 정보로 전략 상태를 업데이트
        
        Args:
            info (list): 거래 정보 리스트
        """
        if not self.is_intialized or info is None:
            return
        
        try:
            # primary_candle 찾기
            target = None
            for item in info:
                if item["type"] == "primary_candle":
                    target = item
                    break
            
            if target is None:
                return
            
            # 데이터 추가
            self.data.append(copy.deepcopy(target))
            
            # 기술적 지표 업데이트
            self._update_rsi(float(target["closing_price"]))
            self._update_volatility()
            self._update_atr(target)
            self._update_bollinger_bands()
            self._update_moving_averages()
            self._update_market_condition()
            self._update_dynamic_rsi_thresholds()
            self._update_position()
            
            # 차트 시각화 업데이트
            self._update_visualization(target)
        
        except Exception as e:
            self.logger.error(f"Error in update_trading_info: {e}")
    
    def _update_visualization(self, target):
        """차트 시각화를 위한 데이터 업데이트"""
        try:
            if self.add_line_callback is not None and len(self.rsi) > 0:
                # RSI 값 표시
                self.add_line_callback(target["date_time"], self.rsi[-1])
                
                # 볼린저 밴드 위치 표시 (0-100 스케일로 변환)
                if self.bb_upper is not None and self.bb_lower is not None:
                    current_price = float(target["closing_price"])
                    price_range = self.bb_upper - self.bb_lower
                    if price_range > 0:
                        # 현재 가격의 볼린저 밴드 내 위치를 0-100 스케일로 변환
                        normalized_position = ((current_price - self.bb_lower) / price_range) * 100
                        normalized_position = max(0, min(normalized_position, 100))  # 0-100 범위로 제한
                        if self.add_spot_callback is not None:
                            self.add_spot_callback(target["date_time"], normalized_position)
        except Exception as e:
            self.logger.error(f"Error in _update_visualization: {e}")
    
    def _update_position(self):
        """현재 가격과 진입가 기반으로 포지션 업데이트"""
        try:
            if len(self.data) < 2:
                return
            
            # 포지션 업데이트 - 현재 종가와 진입가 비교
            if self.entry_price is not None:
                self.position = "buy" if self.entry_price > float(self.data[-1]["closing_price"]) else "sell"
            else:
                # RSI 기반 포지션 결정
                if len(self.rsi) > 0:
                    if self.rsi[-1] < self.rsi_dynamic_low:
                        self.position = "buy"
                        self.logger.debug(f"[RSI-A] Update position to BUY {self.rsi[-1]:.2f} (threshold: {self.rsi_dynamic_low:.2f})")
                    elif self.rsi[-1] > self.rsi_dynamic_high:
                        self.position = "sell"
                        self.logger.debug(f"[RSI-A] Update position to SELL {self.rsi[-1]:.2f} (threshold: {self.rsi_dynamic_high:.2f})")
        except Exception as e:
            self.logger.error(f"Error in _update_position: {e}")
            self.position = None
    
    def _update_rsi(self, price):
        """RSI 지표 업데이트"""
        try:
            price = float(price)
            
            # 필요 갯수만큼 데이터 채우기
            if len(self.rsi) < self.RSI_COUNT:
                self.logger.debug(f"[RSI-A] Fill to ready {price}")
                self.rsi.append(price)
                return
            
            # 초기값 생성
            if len(self.rsi) == self.RSI_COUNT:
                self.logger.debug(f"[RSI-A] Make seed {price}")
                self.rsi.append(price)
                deltas = np.diff(self.rsi)
                
                # 상승, 하락 평균 계산
                up_deltas = deltas.copy()
                up_deltas[up_deltas < 0] = 0
                up_avg = np.mean(up_deltas)
                
                down_deltas = deltas.copy()
                down_deltas[down_deltas > 0] = 0
                down_avg = -np.mean(down_deltas)
                
                # 안전한 나눗셈을 위한 체크
                r_strength = up_avg / down_avg if down_avg != 0 else float('inf')
                
                # RSI 정보 저장
                self.rsi_info = (down_avg, up_avg, price)
                
                # 이전 데이터를 모두 같은 RSI로 설정 (차트 표시용)
                rsi_value = 100.0 - (100.0 / (1.0 + r_strength))
                for i in range(len(self.rsi)):
                    self.rsi[i] = rsi_value
                return
            
            # 최신 RSI 업데이트
            if self.rsi_info is not None:
                up_val = 0.0
                down_val = 0.0
                delta = price - self.rsi_info[2]
                
                if delta > 0:
                    up_val = delta
                else:
                    down_val = -delta
                
                down_avg = (self.rsi_info[0] * (self.RSI_COUNT - 1) + down_val) / self.RSI_COUNT
                up_avg = (self.rsi_info[1] * (self.RSI_COUNT - 1) + up_val) / self.RSI_COUNT
                
                # 안전한 나눗셈
                r_strength = up_avg / down_avg if down_avg != 0 else float('inf')
                
                # RSI 값 계산 및 저장
                rsi_value = 100.0 - (100.0 / (1.0 + r_strength))
                self.rsi.append(rsi_value)
                
                # 최근 값으로 업데이트
                self.rsi_info = (down_avg, up_avg, price)
                self.logger.debug(f"[RSI-A] Update RSI {self.rsi[-1]:.2f}")
        except Exception as e:
            self.logger.error(f"Error in _update_rsi: {e}")
    
    def _update_volatility(self):
        """가격 데이터 기반 변동성 업데이트"""
        try:
            if len(self.data) < self.VOLATILITY_PERIOD + 1:
                return
            
            # 종가 데이터 추출
            try:
                closing_prices = np.array([float(item["closing_price"]) for item in self.data[-self.VOLATILITY_PERIOD:]])
            except (ValueError, KeyError):
                self.logger.error("Invalid closing price data for volatility calculation")
                return
            
            if len(closing_prices) < 2:
                return
            
            # 일별 수익률 계산
            returns = np.diff(closing_prices) / closing_prices[:-1]
            returns = returns[~np.isnan(returns) & ~np.isinf(returns)]  # nan, inf 제거
            
            if len(returns) > 0:
                # 변동성 = 일별 수익률의 표준편차
                self.volatility = float(np.std(returns))
                self.logger.debug(f"[RSI-A] Updated volatility: {self.volatility:.4f}")
            else:
                self.volatility = 0.01  # 기본값
        except Exception as e:
            self.logger.error(f"Error in _update_volatility: {e}")
            self.volatility = 0.01  # 오류 시 기본값
    
    def _update_atr(self, new_price_info):
        """ATR(Average True Range) 업데이트"""
        try:
            if len(self.data) < 2:
                return
            
            # 이전 캔들 정보
            prev_close = float(self.data[-2]["closing_price"])
            
            # 현재 캔들 정보
            current_high = float(new_price_info["high_price"])
            current_low = float(new_price_info["low_price"])
            current_close = float(new_price_info["closing_price"])
            
            # True Range 계산: 세 값 중 가장 큰 값
            true_range = max(
                current_high - current_low,
                abs(current_high - prev_close),
                abs(current_low - prev_close)
            )
            
            self.atr_values.append(true_range)
            
            # ATR 계산 기간에 맞게 데이터 유지
            if len(self.atr_values) > self.atr_period:
                self.atr_values.pop(0)
            
            # ATR 계산 (단순 이동평균)
            if len(self.atr_values) == self.atr_period:
                self.current_atr = np.mean(self.atr_values)
                self.logger.debug(f"[RSI-A] Updated ATR: {self.current_atr:.4f}")
        except Exception as e:
            self.logger.error(f"Error in _update_atr: {e}")
    
    def _update_bollinger_bands(self):
        """볼린저 밴드 업데이트"""
        try:
            if len(self.data) < self.BB_PERIOD:
                return
            
            # 종가 데이터 추출
            closing_prices = np.array([float(item["closing_price"]) for item in self.data[-self.BB_PERIOD:]])
            
            # 중간 밴드 (20일 이동평균)
            self.bb_middle = np.mean(closing_prices)
            
            # 표준편차 계산
            std_dev = np.std(closing_prices)
            
            # 상단 및 하단 밴드 계산
            self.bb_upper = self.bb_middle + (self.BB_STD_DEV * std_dev)
            self.bb_lower = self.bb_middle - (self.BB_STD_DEV * std_dev)
            
            self.logger.debug(f"[RSI-A] BB: Middle={self.bb_middle:.2f}, Upper={self.bb_upper:.2f}, Lower={self.bb_lower:.2f}")
        except Exception as e:
            self.logger.error(f"Error in _update_bollinger_bands: {e}")
    
    def _update_moving_averages(self):
        """이동평균선 업데이트"""
        try:
            if len(self.data) == 0:
                return
            
            # 단기 이동평균
            if len(self.data) >= self.MA_SHORT_PERIOD:
                prices = [float(item["closing_price"]) for item in self.data[-self.MA_SHORT_PERIOD:]]
                self.ma_short = np.mean(prices)
            
            # 중기 이동평균
            if len(self.data) >= self.MA_MEDIUM_PERIOD:
                prices = [float(item["closing_price"]) for item in self.data[-self.MA_MEDIUM_PERIOD:]]
                self.ma_medium = np.mean(prices)
            
            # 장기 이동평균
            if len(self.data) >= self.MA_LONG_PERIOD:
                prices = [float(item["closing_price"]) for item in self.data[-self.MA_LONG_PERIOD:]]
                self.ma_long = np.mean(prices)
            
            # 이동평균 추세 판단
            if self.ma_short is not None and self.ma_medium is not None:
                if self.ma_short > self.ma_medium:
                    if self.ma_long is not None and self.ma_medium > self.ma_long:
                        self.ma_trend = "strong_bullish"  # 골든크로스 상태
                    else:
                        self.ma_trend = "bullish"
                elif self.ma_short < self.ma_medium:
                    if self.ma_long is not None and self.ma_medium < self.ma_long:
                        self.ma_trend = "strong_bearish"  # 데드크로스 상태
                    else:
                        self.ma_trend = "bearish"
                else:
                    self.ma_trend = "neutral"
                
                self.logger.debug(f"[RSI-A] MA Trend: {self.ma_trend}")
        except Exception as e:
            self.logger.error(f"Error in _update_moving_averages: {e}")
    
    def _update_market_condition(self):
        """시장 상태 업데이트 (추세/횡보)"""
        try:
            if len(self.data) < self.VOLATILITY_PERIOD or self.volatility is None:
                self.market_condition = "unknown"
                return
            
            # 종가 데이터 추출
            try:
                closing_prices = np.array([float(item["closing_price"]) for item in self.data[-self.VOLATILITY_PERIOD:]])
            except (ValueError, KeyError):
                self.logger.error("Invalid closing price data for market condition analysis")
                self.market_condition = "unknown"
                return
            
            if len(closing_prices) < 3:
                self.market_condition = "unknown"
                return
            
            # 가격 방향 계산
            price_direction = np.sign(np.diff(closing_prices))
            
            # 방향 변화 횟수 계산
            direction_changes = np.sum(np.abs(np.diff(price_direction)))
            
            # 방향 변화가 적고 변동성이 중간 이상이면 추세장, 그렇지 않으면 횡보장
            if direction_changes < self.VOLATILITY_PERIOD / 3 and self.volatility > 0.01:
                self.market_condition = "trend"
            else:
                self.market_condition = "range"
            
            self.logger.debug(f"[RSI-A] Market condition: {self.market_condition} (changes: {direction_changes}, volatility: {self.volatility:.4f})")
        except Exception as e:
            self.logger.error(f"Error in _update_market_condition: {e}")
            self.market_condition = "unknown"
    
    def _update_dynamic_rsi_thresholds(self):
        """변동성 기반 RSI 기준점 동적 조정"""
        try:
            if len(self.data) < self.VOLATILITY_PERIOD or self.volatility is None:
                # 기본값 사용
                self.rsi_dynamic_low = self.RSI_BASE_LOW
                self.rsi_dynamic_high = self.RSI_BASE_HIGH
                self.risk_ratio = 1.0
                return
            
            # 변동성에 기반한 조정 계수 (최대 10으로 제한)
            volatility_factor = min(int(self.volatility * 100), 10)
            
            # RSI 기준점 조정
            self.rsi_dynamic_low = max(self.RSI_BASE_LOW - volatility_factor, 20)
            self.rsi_dynamic_high = min(self.RSI_BASE_HIGH + volatility_factor, 80)
            
            # 시장 상태에 따른 리스크 비율 조정
            if self.market_condition == "trend":
                # 추세장에서는 더 적극적으로 매매 (리스크 증가)
                self.risk_ratio = min(1.0, 0.6 + self.volatility * 2)
            else:
                # 횡보장에서는 보수적으로 매매 (리스크 감소)
                self.risk_ratio = max(0.5, 0.7 - self.volatility * 2)
            
            self.logger.debug(f"[RSI-A] Dynamic thresholds: Low={self.rsi_dynamic_low:.1f}, High={self.rsi_dynamic_high:.1f}, Risk={self.risk_ratio:.2f}")
        except Exception as e:
            self.logger.error(f"Error in _update_dynamic_rsi_thresholds: {e}")
            # 오류 시 기본값 사용
            self.rsi_dynamic_low = self.RSI_BASE_LOW
            self.rsi_dynamic_high = self.RSI_BASE_HIGH
            self.risk_ratio = 1.0
    
    def _is_good_entry_point(self):
        """매매 진입점 적합성 판단"""
        try:
            if len(self.data) < max(self.MA_SHORT_PERIOD, self.BB_PERIOD):
                return True  # 충분한 데이터가 없으면 기본 신호 사용
            
            current_price = float(self.data[-1]["closing_price"])
            
            # 현재 변동성 체크 - 변동성이 매우 높으면 거래 제한
            if self.volatility is not None and self.volatility > 0.05:
                self.logger.info(f"[FILTER] High volatility: {self.volatility:.4f}, skipping trade")
                return False
            
            # 볼린저 밴드 위치 체크
            bb_condition = False
            if self.bb_lower is not None and self.bb_upper is not None:
                # 현재 가격의 볼린저 밴드 내 위치를 백분율로 계산 (0-100%)
                bb_percent = (current_price - self.bb_lower) / (self.bb_upper - self.bb_lower) if self.bb_upper != self.bb_lower else 0.5
                
                # 매수 시그널에서는 가격이 하단에 가까울수록 좋음 (0-30%)
                if self.position == "buy" and bb_percent < 0.3:
                    bb_condition = True
                # 매도 시그널에서는 가격이 상단에 가까울수록 좋음 (70-100%)
                elif self.position == "sell" and bb_percent > 0.7:
                    bb_condition = True
            
            # 이동평균선 추세 체크
            ma_condition = False
            if self.position == "buy":
                ma_condition = self.ma_trend in ["bullish", "strong_bullish"]
            elif self.position == "sell":
                ma_condition = self.ma_trend in ["bearish", "strong_bearish"]
            
            # 물타기 방지를 위한 추가 조건
            trend_relief = False
            if len(self.data) >= 3 and self.position == "buy":
                recent_prices = [float(d["closing_price"]) for d in self.data[-3:]]
                # 하락 속도가 완화되는 경우
                if (recent_prices[2] - recent_prices[1]) > (recent_prices[1] - recent_prices[0]):
                    trend_relief = True
            
            # 모든 조건을 AND로 결합하여 진입 판단 (엄격한 필터링)
            return bb_condition and (ma_condition or trend_relief)
        
        except Exception as e:
            self.logger.error(f"Error in _is_good_entry_point: {e}")
            return True  # 오류 시 기본 신호 사용
    
    def update_result(self, result):
        """
        거래 결과 업데이트
        
        Args:
            result (dict): 거래 결과 정보
        """
        if not self.is_intialized:
            return
        
        try:
            request = result["request"]
            
            # 요청 상태 업데이트
            if result["state"] == "requested":
                self.waiting_requests[request["id"]] = result
                return
            
            if result["state"] == "done" and request["id"] in self.waiting_requests:
                del self.waiting_requests[request["id"]]
            
            price = float(result["price"])
            amount = float(result["amount"])
            
            if price <= 0 or amount <= 0:
                return
            
            total = price * amount
            fee = total * self.COMMISSION_RATIO
            
            if result["type"] == "buy":
                self.balance -= round(total + fee)
                
                # 진입 가격 업데이트 (매수 시)
                if result["msg"] == "success" and amount > 0:
                    self.entry_price = price
                    self.last_trade_result = {"buy_price": price, "buy_time": datetime.now()}
                    
                    # ATR 기반 손절/익절 가격 계산
                    if self.current_atr is not None and self.current_atr > 0:
                        self.stop_loss_price = self.entry_price - (self.current_atr * self.sl_atr_factor)
                        self.take_profit_price = self.entry_price + (self.current_atr * self.tp_atr_factor)
                        self.logger.debug(f"[RSI-A] Entry: {self.entry_price:.2f}, SL: {self.stop_loss_price:.2f}, TP: {self.take_profit_price:.2f}")
                    else:
                        # ATR이 없는 경우 기본 비율로 계산
                        self.stop_loss_price = self.entry_price * (1 - self.STOP_LOSS_RATIO)
                        self.take_profit_price = self.entry_price * (1 + self.TAKE_PROFIT_RATIO)
                        self.logger.debug(f"[RSI-A] Entry: {self.entry_price:.2f}, Basic SL: {self.stop_loss_price:.2f}, TP: {self.take_profit_price:.2f}")
                    
                    # 부분 익절 초기화
                    self.partial_tp_executed = False
            else:  # sell
                self.balance += round(total - fee)
                
                # 매도 결과 처리
                if result["msg"] == "success":
                    if self.last_trade_result is not None and "buy_price" in self.last_trade_result:
                        self.last_trade_result["sell_price"] = price
                        self.last_trade_result["sell_time"] = datetime.now()
                        
                        # 손익 계산
                        profit_ratio = (price / self.last_trade_result["buy_price"]) - 1
                        self.logger.info(f"[RESULT] Trade completed with {profit_ratio*100:.2f}% profit")
                        
                        # 전량 매도 시 상태 초기화
                        if self.asset_amount - amount <= 0:
                            self.entry_price = None
                            self.stop_loss_price = None
                            self.take_profit_price = None
                            self.partial_tp_executed = False
            
            # 자산 수량 업데이트
            if result["msg"] == "success":
                if result["type"] == "buy":
                    self.asset_amount = round(self.asset_amount + amount, 6)
                elif result["type"] == "sell":
                    self.asset_amount = round(self.asset_amount - amount, 6)
                
                # 자산이 미미한 경우 0으로 설정
                if self.asset_amount < 0.000001:
                    self.asset_amount = 0
            
            # 결과 로깅
            self.logger.info(f"[RESULT] id: {result['request']['id']} ================")
            self.logger.info(f"type: {result['type']}, msg: {result['msg']}")
            self.logger.info(f"price: {price}, amount: {amount}")
            self.logger.info(f"balance: {self.balance}, asset_amount: {self.asset_amount}")
            
            if self.entry_price is not None:
                current_price = float(self.data[-1]["closing_price"]) if self.data else 0
                price_change = ((current_price / self.entry_price) - 1) * 100
                self.logger.info(f"entry_price: {self.entry_price}, change: {price_change:.2f}%")
            
            self.logger.info("================================================")
            self.result.append(copy.deepcopy(result))
        
        except (AttributeError, TypeError, KeyError, ValueError) as msg:
            self.logger.error(f"Error updating result: {msg}")
        except Exception as e:
            self.logger.error(f"Unexpected error in update_result: {e}")
    
    def __create_buy(self, price, amount=0):
        """매수 요청 생성"""
        try:
            if price <= 0:
                self.logger.error(f"Invalid price for buy: {price}")
                return None
            
            # 현재 시간 설정
            now = datetime.now().strftime(self.ISO_DATEFORMAT)
            if self.is_simulation:
                now = self.data[-1]["date_time"] if self.data else now
            
            # 매수 예산 계산 (잔고 * 리스크 비율)
            buy_budget = self.balance * self.risk_ratio
            
            # 거래 시스템 최대 거래 금액 제한
            MAX_TRADE_VALUE = 1000000.0
            buy_budget = min(buy_budget, MAX_TRADE_VALUE)
            
            # 최소 거래 금액 적용
            if buy_budget < self.min_price and self.balance >= self.min_price:
                buy_budget = self.min_price
            elif buy_budget < self.min_price:
                self.logger.debug(f"[RSI-A] Budget ({buy_budget:.2f}) less than min_price ({self.min_price})")
                if self.is_simulation:
                    return {
                        "id": DateConverter.timestamp_id(),
                        "type": "buy",
                        "price": 0,
                        "amount": 0,
                        "date_time": now,
                    }
                return None
            
            # 요청 수량 계산
            req_amount = buy_budget / price
            
            # 소수점 4자리 아래 버림
            req_amount = math.floor(req_amount * 10000) / 10000
            
            # 최종 거래 금액 확인
            final_value = req_amount * price
            
            if final_value < self.min_price:
                self.logger.debug(f"[RSI-A] Final value ({final_value:.2f}) less than min_price ({self.min_price})")
                if self.is_simulation:
                    return {
                        "id": DateConverter.timestamp_id(),
                        "type": "buy",
                        "price": 0,
                        "amount": 0,
                        "date_time": now,
                    }
                return None
            
            # 유효한 요청 반환
            return {
                "id": DateConverter.timestamp_id(),
                "type": "buy",
                "price": price,
                "amount": req_amount,
                "date_time": now,
            }
        
        except Exception as e:
            self.logger.error(f"Error in __create_buy: {e}")
            if self.is_simulation:
                return {
                    "id": DateConverter.timestamp_id(),
                    "type": "buy",
                    "price": 0,
                    "amount": 0,
                    "date_time": datetime.now().strftime(self.ISO_DATEFORMAT),
                }
            return None
    
    def __create_sell(self, price, amount):
        """매도 요청 생성"""
        try:
            if price <= 0:
                self.logger.error(f"Invalid price for sell: {price}")
                return None
            
            req_amount = float(amount)
            
            # 요청 수량이 보유 수량보다 큰 경우 조정
            if req_amount > self.asset_amount:
                req_amount = self.asset_amount
            
            # 소숫점 4자리 아래 버림
            req_amount = math.floor(req_amount * 10000) / 10000
            
            # 총 거래 금액 계산
            total_value = price * req_amount
            
            # 유효성 검사
            if req_amount <= 0 or total_value < self.min_price:
                self.logger.info(f"Asset too small: amount={req_amount}, value={total_value:.2f}")
                if self.is_simulation:
                    now = self.data[-1]["date_time"] if self.data else datetime.now().strftime(self.ISO_DATEFORMAT)
                    return {
                        "id": DateConverter.timestamp_id(),
                        "type": "sell",
                        "price": 0,
                        "amount": 0,
                        "date_time": now,
                    }
                return None
            
            # 현재 시간 설정
            now = datetime.now().strftime(self.ISO_DATEFORMAT)
            if self.is_simulation:
                now = self.data[-1]["date_time"] if self.data else now
            
            return {
                "id": DateConverter.timestamp_id(),
                "type": "sell",
                "price": price,
                "amount": req_amount,
                "date_time": now,
            }
        
        except Exception as e:
            self.logger.error(f"Error in __create_sell: {e}")
            if self.is_simulation:
                return {
                    "id": DateConverter.timestamp_id(),
                    "type": "sell",
                    "price": 0,
                    "amount": 0,
                    "date_time": datetime.now().strftime(self.ISO_DATEFORMAT),
                }
            return None
    
    def __create_partial_take_profit(self, price):
        """부분 익절 요청 생성"""
        try:
            if self.asset_amount <= 0:
                return None
            
            # 판매할 수량 계산 (자산의 50%)
            partial_amount = self.asset_amount * self.PARTIAL_TP_RATIO
            
            # 소수점 4자리 아래 버림
            partial_amount = math.floor(partial_amount * 10000) / 10000
            
            # 총 거래 금액 계산
            total_value = price * partial_amount
            
            # 최소 거래 금액 확인
            if partial_amount <= 0 or total_value < self.min_price:
                self.logger.info(f"부분 익절 금액이 너무 작음: {partial_amount}, 가치={total_value:.2f}")
                return None
            
            # 현재 시간 설정
            now = datetime.now().strftime(self.ISO_DATEFORMAT)
            if self.is_simulation:
                now = self.data[-1]["date_time"] if self.data else now
            
            # 취소 요청과 부분 매도 요청 생성
            final_requests = []
            
            # 기존 요청 취소
            for request_id in self.waiting_requests:
                final_requests.append({
                    "id": request_id,
                    "type": "cancel",
