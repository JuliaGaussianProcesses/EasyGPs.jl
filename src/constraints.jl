using ParameterHandling: AbstractParameter, ParameterHandling
import ParameterHandling: flatten, value

struct RealParameter{T<:Real,Tp<:AbstractParameter} <: Real
    par::Tp
end

# `ParameterHandling.value` is overloaded to keep `RealParameter`s untouched
value(p::RealParameter) = p

# A custom version of `ParameterHandling.value` that forwards `RealParameter`s and otherwise
# has the same behavior.
_value(p::AbstractParameter) = value(p)
_value(p::RealParameter) = value(p.par)
_value(x::Number) = x
_value(x::AbstractArray{<:Number}) = x
_value(x::AbstractArray) = map(_value, x)
_value(x::Tuple) = map(_value, x)
_value(x::NamedTuple) = map(_value, x)
_value(x::Dict) = Dict(k => _value(v) for (k, v) in x)
_value(::Nothing) = nothing

_isequal(p1::T, p2::T) where T<:RealParameter = _value(p1) ≈ _value(p2)

function flatten(::Type{T1}, p::RealParameter{T2}) where {T1<:Real,T2<:Real}
    v, unflatten = flatten(T1, p.par)
    function unflatten_to_RealParameter(v::Vector{T1})
        return RealParameter{T2,typeof(p.par)}(unflatten(v))
    end
    return v, unflatten_to_RealParameter
end

Base.convert(::Type{<:RealParameter}, x::RealParameter) = x
Base.convert(::Type{T}, x::RealParameter) where {T<:Real} = convert(T, _value(x))
Base.promote(x::RealParameter, y::Number) = promote(_value(x), y)
Base.promote(x::Number, y::RealParameter) = reverse(promote(y, x))
Base.zero(::Type{<:RealParameter{T}}) where {T} = zero(T)

function Base.inv(p::RealParameter{T,<:ParameterHandling.Fixed}) where {T<:Real}
    return fixed(inv(_value(p)))
end

function positive(x::Real)
    _x = ParameterHandling.positive(x)
    return RealParameter{typeof(x),typeof(_x)}(_x)
end

ParameterHandling.positive(p::RealParameter) = p

function fixed(x::Real)
    _x = ParameterHandling.fixed(x)
    return RealParameter{typeof(x),typeof(_x)}(_x)
end

ParameterHandling.fixed(p::RealParameter) = p
