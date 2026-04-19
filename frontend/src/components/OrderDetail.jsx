function OrderDetail({ order, driver, t }) {
  const renderOrderStatus = (statusCode) => {
    return t.orderStatus[statusCode] ?? statusCode;
  };

  return (
    <div className="panel order-detail">
      <h2>{t.orderDetailTitle}</h2>

      {!order && <div>{t.selectOrderHint}</div>}

      {order && (
        <div>
          <p>{t.passenger}：{order.passengerName}</p>
          <p>{t.pickup}：{order.pickupAddress}</p>
          <p>{t.dropoff}：{order.dropoffAddress}</p>
          <p>{t.status}：{renderOrderStatus(order.status)}</p>
          <p>
            {t.assignedDriver}：
            {driver ? driver.name : t.notAssigned}
          </p>
        </div>
      )}
    </div>
  );
}

export default OrderDetail;
