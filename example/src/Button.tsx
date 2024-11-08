import React from 'react'
import { Text, StyleSheet, StyleProp, ViewStyle } from 'react-native'
import { TouchableOpacity } from 'react-native-gesture-handler'

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#333',
    borderRadius: 5,
    padding: 8,
    margin: 4,
    width: 'auto',
  },
  buttonText: {
    color: '#ccc',
    fontSize: 12,
    fontWeight: 'bold',
  },
})

export function Button({
  style,
  title,
  onPress,
  disabled,
}: {
  style?: StyleProp<ViewStyle>
  title: string
  onPress: () => void
  disabled?: boolean
}) {
  return (
    <TouchableOpacity
      style={[styles.button, style]}
      onPress={onPress}
      disabled={disabled}
    >
      <Text style={styles.buttonText}>{title}</Text>
    </TouchableOpacity>
  )
}

Button.defaultProps = {
  style: undefined,
  disabled: false,
}
